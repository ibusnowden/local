/**
 * State, persistence, and formatting helpers for the /goal evaluator loop.
 *
 * This module is intentionally framework-light: it holds the data types and
 * pure helpers used by index.ts (orchestration) and evaluator.ts (subprocess).
 */
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";

/** Session-entry customType used to persist goal state across reloads/resumes. */
export const GOAL_STATE_TYPE = "goal-state";

/** Custom message type used for goal notifications shown in the conversation. */
export const GOAL_MESSAGE_TYPE = "goal";

export interface GoalConfig {
	/** Max evaluator iterations per active goal period. */
	maxIterations: number;
	/** Model pattern/id for the evaluator subprocess, or undefined to use the default model. */
	evaluatorModel: string | undefined;
	/** Built-in tools the evaluator subprocess may use (read-only by default). */
	evaluatorTools: string[];
}

export interface GoalState {
	goalText: string;
	active: boolean;
	iteration: number;
	config: GoalConfig;
}

export interface EvaluationResult {
	achieved: boolean;
	confidence: "high" | "medium" | "low";
	summary: string;
	gaps: string[];
	nextAction: string;
	/** Raw evaluator text (for debugging / details). */
	raw: string;
	/** Set when the evaluator failed to produce a usable verdict. */
	error?: string;
	/** Token usage of the evaluator subprocess (last assistant message). */
	usage?: { turns: number; input: number; output: number; cost: number };
}

export const DEFAULT_CONFIG: GoalConfig = {
	maxIterations: 10,
	evaluatorModel: undefined,
	evaluatorTools: ["read", "grep", "find", "ls"],
};

export function cloneConfig(c: GoalConfig): GoalConfig {
	return {
		maxIterations: c.maxIterations,
		evaluatorModel: c.evaluatorModel,
		evaluatorTools: [...c.evaluatorTools],
	};
}

export function defaultState(): GoalState {
	return { goalText: "", active: false, iteration: 0, config: cloneConfig(DEFAULT_CONFIG) };
}

/** Persist the current goal state as a custom session entry (not sent to the LLM). */
export function persistGoal(pi: ExtensionAPI, state: GoalState): void {
	pi.appendEntry(GOAL_STATE_TYPE, {
		goalText: state.goalText,
		active: state.active,
		iteration: state.iteration,
		config: cloneConfig(state.config),
	});
}

/**
 * Reconstruct goal state from session entries.
 *
 * The most recent goal-state entry wins. `active` is always restored as false
 * so that reloading or resuming a session never auto-starts the evaluator loop;
 * the user must explicitly `/goal continue` to resume.
 */
export function restoreGoalFromEntries(entries: SessionEntry[]): GoalState {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type === "custom" && entry.customType === GOAL_STATE_TYPE) {
			const data = (entry as { data?: Partial<GoalState> }).data;
			if (data) {
				return {
					goalText: typeof data.goalText === "string" ? data.goalText : "",
					active: false,
					iteration: typeof data.iteration === "number" ? data.iteration : 0,
					config: {
						maxIterations: data.config?.maxIterations ?? DEFAULT_CONFIG.maxIterations,
						evaluatorModel: data.config?.evaluatorModel ?? DEFAULT_CONFIG.evaluatorModel,
						evaluatorTools: data.config?.evaluatorTools ?? [...DEFAULT_CONFIG.evaluatorTools],
					},
				};
			}
		}
	}
	return defaultState();
}

/* -------------------------------------------------------------------------- */
/* Transcript building (for the evaluator subprocess)                         */
/* -------------------------------------------------------------------------- */

export interface TranscriptOptions {
	maxMessages?: number;
	maxCharsPerMessage?: number;
	maxTotalChars?: number;
}

/**
 * Build a compact text transcript of the recent session for the evaluator.
 * Takes the last N messages from the branch and formats them, capped to a
 * total character budget (oldest messages dropped first).
 */
export function buildTranscript(entries: SessionEntry[], options: TranscriptOptions = {}): string {
	const maxMessages = options.maxMessages ?? 30;
	const maxCharsPerMessage = options.maxCharsPerMessage ?? 1200;
	const maxTotalChars = options.maxTotalChars ?? 20000;

	const messages: unknown[] = [];
	for (const entry of entries) {
		if (entry.type === "message") {
			messages.push((entry as { message: unknown }).message);
		}
	}

	const recent = messages.slice(-maxMessages);

	const blocks: string[] = [];
	for (const msg of recent) {
		const block = formatMessageForTranscript(msg, maxCharsPerMessage);
		if (block) blocks.push(block);
	}

	// Trim oldest blocks until under the total budget.
	while (blocks.length > 1 && blocks.join("\n\n---\n\n").length > maxTotalChars) {
		blocks.shift();
	}

	if (blocks.length === 0) {
		return "(no prior conversation)";
	}
	return blocks.join("\n\n---\n\n");
}

function formatMessageForTranscript(msg: unknown, maxChars: number): string {
	if (!msg || typeof msg !== "object") return "";
	const role = (msg as { role?: string }).role;

	if (role === "user") {
		const text = extractTextContent((msg as { content?: unknown }).content);
		if (!text) return "";
		return `## User\n${truncate(text, maxChars)}`;
	}

	if (role === "assistant") {
		const text = extractAssistantText(msg);
		const tools = extractToolCallSummary(msg);
		let body = text;
		if (tools) body = body ? `${body}\n${tools}` : tools;
		if (!body) return "";
		return `## Assistant\n${truncate(body, maxChars)}`;
	}

	if (role === "toolResult") {
		const m = msg as { toolName?: string; isError?: boolean; content?: unknown };
		const text = extractTextContent(m.content);
		const name = m.toolName ?? "tool";
		const errorTag = m.isError ? " (error)" : "";
		return `### Tool result${errorTag}: ${name}\n${truncate(text, maxChars) || "(no output)"}`;
	}

	// Skip custom, bashExecution, branchSummary, compactionSummary, etc.
	return "";
}

function extractTextContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((c): c is { type: "text"; text: string } => !!c && typeof c === "object" && (c as { type?: string }).type === "text")
		.map((c) => c.text)
		.join("\n");
}

function extractAssistantText(msg: unknown): string {
	const content = (msg as { content?: unknown }).content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((c): c is { type: "text"; text: string } => !!c && typeof c === "object" && (c as { type?: string }).type === "text")
		.map((c) => c.text)
		.join("\n");
}

function extractToolCallSummary(msg: unknown): string {
	const content = (msg as { content?: unknown }).content;
	if (!Array.isArray(content)) return "";
	const calls = content.filter(
		(c): c is { type: "toolCall"; name: string; arguments: Record<string, unknown> } =>
			!!c && typeof c === "object" && (c as { type?: string }).type === "toolCall",
	);
	if (calls.length === 0) return "";
	return `Tools: ${calls.map((c) => `${c.name}(${briefArgs(c.arguments)})`).join(", ")}`;
}

function briefArgs(args: Record<string, unknown> | undefined): string {
	if (!args || typeof args !== "object") return "";
	const keys = Object.keys(args).slice(0, 2);
	return keys
		.map((k) => `${k}=${shortRepr(args[k])}`)
		.join(", ");
}

function shortRepr(value: unknown): string {
	if (typeof value === "string") return JSON.stringify(value.length > 40 ? `${value.slice(0, 40)}…` : value);
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	return "…";
}

function truncate(text: string, max: number): string {
	if (text.length <= max) return text;
	return `${text.slice(0, max)}…`;
}

/* -------------------------------------------------------------------------- */
/* Message formatting helpers (kickoff, feedback, notifications)              */
/* -------------------------------------------------------------------------- */

export function formatKickoff(goalText: string, maxIterations: number): string {
	return [
		"I have set a goal for you to work toward autonomously.",
		"",
		`Goal: ${goalText}`,
		"",
		"Work toward this goal using your available tools. After each step, an evaluator will review your progress and give you specific feedback on what remains. Continue autonomously until the evaluator confirms the goal is fully achieved.",
		"",
		"Guidelines:",
		"- Do not ask me questions; make reasonable assumptions and proceed.",
		"- Do not repeat work you have already completed.",
		"- Use the goal_status tool if you need to recall the goal or current iteration.",
		"- When you believe the goal is achieved, briefly summarize what you accomplished; the evaluator will verify.",
		`- The evaluator loop runs for up to ${maxIterations} iterations.`,
	].join("\n");
}

export function formatResume(goalText: string, iteration: number, maxIterations: number): string {
	return [
		"Resuming work on the goal.",
		"",
		`Goal: ${goalText}`,
		`Iteration: ${iteration}/${maxIterations}`,
		"",
		"Continue where you left off. The evaluator will review your progress after this turn.",
	].join("\n");
}

export function formatFeedback(
	goalText: string,
	result: EvaluationResult,
	iteration: number,
	maxIterations: number,
	generic: boolean,
): string {
	const lines: string[] = [];
	lines.push(`[Evaluator — iteration ${iteration}/${maxIterations}: goal not yet achieved]`);
	if (result.summary) lines.push(`Summary: ${result.summary.trim()}`);
	if (result.gaps.length > 0) {
		lines.push("Remaining gaps:");
		for (const gap of result.gaps) lines.push(`- ${gap}`);
	}
	if (generic || !result.nextAction.trim()) {
		lines.push("Continue working toward the goal. Do not repeat completed work.");
	} else {
		lines.push(`Next action: ${result.nextAction.trim()}`);
		lines.push("Continue working toward the goal. Do not repeat completed work.");
	}
	lines.push("");
	lines.push(`Goal: ${goalText}`);
	return lines.join("\n");
}

export function formatGoalForAgent(state: GoalState): string {
	if (!state.goalText) return "No goal is currently set.";
	const status = state.active ? "active (evaluator loop running)" : "paused";
	return [
		`Current goal: ${state.goalText}`,
		`Iteration: ${state.iteration}/${state.config.maxIterations}`,
		`Status: ${status}`,
		state.config.evaluatorModel ? `Evaluator model: ${state.config.evaluatorModel}` : `Evaluator model: default`,
	].join("\n");
}
