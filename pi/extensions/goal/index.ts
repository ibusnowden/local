/**
 * /goal — autonomous goal with an evaluator loop.
 *
 * Commands:
 *   /goal <description>   Set a goal and start the evaluator loop.
 *   /goal set <text>      Explicit form for setting a goal (use when the text
 *                         starts with a subcommand name like "stop").
 *   /goal status          Show the current goal, iteration, and config.
 *   /goal continue        Resume a paused goal.
 *   /goal stop            Stop the evaluator loop (keeps the goal paused).
 *   /goal clear           Clear the goal entirely.
 *   /goal config          Show / change evaluator config.
 *   /goal help            Show help.
 *
 * How it works:
 *   - /goal frames the goal and sends a kickoff user message.
 *   - After each agent turn (agent_end), a read-only evaluator subprocess
 *     inspects the codebase + transcript and emits a JSON verdict.
 *   - If the goal is not yet achieved, the verdict is injected as a follow-up
 *     user message so the agent keeps working. This repeats until the goal is
 *     achieved, the iteration cap is reached, the agent is aborted, or the
 *     user runs /goal stop.
 *   - State persists across /reload and session resume (always paused; use
 *     /goal continue to resume).
 */
import type {
	AgentEndEvent,
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import {
	buildTranscript,
	defaultState,
	formatFeedback,
	formatGoalForAgent,
	formatKickoff,
	formatResume,
	GOAL_MESSAGE_TYPE,
	GOAL_STATE_TYPE,
	persistGoal,
	restoreGoalFromEntries,
	type EvaluationResult,
	type GoalState,
} from "./state.ts";
import { runEvaluator } from "./evaluator.ts";

export default function goalExtension(pi: ExtensionAPI): void {
	let goal: GoalState = defaultState();
	/** True while an evaluator subprocess is running (guards re-entrancy). */
	let evaluating = false;
	/** Set when /goal stop/clear or a new goal aborts an in-flight evaluator. */
	let stopRequested = false;
	/** Abort controller for the current evaluator subprocess. */
	let evaluatorAbort: AbortController | null = null;
	/** Number of consecutive evaluator failures (reset on success). */
	let consecutiveErrors = 0;
	/**
	 * Bumped whenever the goal is replaced, stopped, cleared, or resumed. The
	 * agent_end handler captures this before awaiting the evaluator and bails
	 * out if it changed, so a concurrent /goal command can't corrupt an
	 * in-flight evaluation.
	 */
	let goalEpoch = 0;

	const bumpEpoch = () => {
		goalEpoch++;
	};

	/* ------------------------------------------------------------------ */
	/* UI helpers                                                          */
	/* ------------------------------------------------------------------ */

	function statusText(state: GoalState, phase?: "evaluating" | "active" | "paused", progressLabel?: string): string {
		if (phase === "evaluating") {
			return `🎯 ${state.iteration + 1}/${state.config.maxIterations} evaluating${progressLabel ? `: ${progressLabel}` : ""}`;
		}
		if (state.active) return `🎯 ${state.iteration}/${state.config.maxIterations}`;
		return `🎯 paused`;
	}

	function widgetLines(state: GoalState, phase?: "evaluating" | "active" | "paused", progressLabel?: string): string[] {
		let status: string;
		if (phase === "evaluating") status = progressLabel ? `evaluating: ${progressLabel}` : "evaluating";
		else if (state.active) status = "active";
		else status = "paused";
		return [`🎯 Goal: ${state.goalText}`, `   Iteration: ${state.iteration}/${state.config.maxIterations} — ${status}`];
	}

	function updateStatus(ctx: ExtensionContext, phase?: "evaluating" | "active" | "paused", progressLabel?: string): void {
		if (!goal.goalText) {
			ctx.ui.setStatus("goal", undefined);
			ctx.ui.setWidget("goal", undefined);
			return;
		}
		ctx.ui.setStatus("goal", statusText(goal, phase, progressLabel));
		ctx.ui.setWidget("goal", widgetLines(goal, phase, progressLabel));
	}

	function notify(ctx: ExtensionContext, message: string, type: "info" | "warning" | "error" = "info"): void {
		ctx.ui.notify(message, type);
	}

	function pushGoalMessage(content: string, details?: Record<string, unknown>): void {
		try {
			pi.sendMessage({ customType: GOAL_MESSAGE_TYPE, content, display: true, details }, { deliverAs: "nextTurn" });
		} catch {
			// sendMessage should not throw in normal operation; ignore.
		}
	}

	/* ------------------------------------------------------------------ */
	/* Goal lifecycle helpers                                              */
	/* ------------------------------------------------------------------ */

	function persist(): void {
		persistGoal(pi, goal);
	}

	function pauseGoal(ctx: ExtensionContext, reason: string): void {
		goal.active = false;
		consecutiveErrors = 0;
		stopRequested = false;
		persist();
		updateStatus(ctx, "paused");
		pushGoalMessage(`⏹ Goal paused: ${reason}`);
	}

	function completeGoal(ctx: ExtensionContext, result: EvaluationResult): void {
		const text = goal.goalText;
		goal.goalText = "";
		goal.active = false;
		goal.iteration = 0;
		consecutiveErrors = 0;
		stopRequested = false;
		persist();
		ctx.ui.setStatus("goal", undefined);
		ctx.ui.setWidget("goal", undefined);
		const summary = result.summary || "Goal achieved.";
		pushGoalMessage(`✅ Goal achieved!\n\nGoal: ${text}\n${summary}`, {
			achieved: true,
			summary,
			confidence: result.confidence,
			gaps: result.gaps,
		});
		notify(ctx, "🎯 Goal achieved!", "info");
	}

	function findLastAssistant(messages: AgentMessage[]): AssistantMessage | undefined {
		for (let i = messages.length - 1; i >= 0; i--) {
			const m = messages[i];
			if (m && m.role === "assistant") return m;
		}
		return undefined;
	}

	function injectFeedback(ctx: ExtensionContext, result: EvaluationResult, generic: boolean): void {
		const feedback = formatFeedback(goal.goalText, result, goal.iteration, goal.config.maxIterations, generic);
		const remaining = goal.config.maxIterations - goal.iteration;
		// sendUserMessage at agent_end (agent is idle) triggers a new turn, continuing the loop.
		try {
			pi.sendUserMessage(feedback, { deliverAs: "followUp" });
		} catch {
			// If something goes wrong, notify and pause rather than spin.
			notify(ctx, "Evaluator could not queue the next step; pausing the goal.", "error");
			pauseGoal(ctx, "failed to queue follow-up");
			return;
		}
		notify(
			ctx,
			`Evaluator: not yet achieved (${remaining} iteration${remaining === 1 ? "" : "s"} left)`,
			"info",
		);
	}

	/* ------------------------------------------------------------------ */
	/* Command handlers                                                    */
	/* ------------------------------------------------------------------ */

	async function startGoal(ctx: ExtensionCommandContext, goalText: string): Promise<void> {
		const trimmed = goalText.trim();
		if (!trimmed) {
			notify(ctx, "Goal text cannot be empty.", "error");
			return;
		}

		// Stop any previous loop and prevent its in-flight evaluator from acting.
		bumpEpoch();
		stopRequested = true;
		evaluatorAbort?.abort();
		evaluatorAbort = null;
		goal.active = false; // so the aborted turn's agent_end skips evaluation

		if (!ctx.isIdle()) {
			notify(ctx, "Aborting current task to start the goal…", "info");
			ctx.abort();
			await ctx.waitForIdle();
		}

		stopRequested = false;
		consecutiveErrors = 0;
		goal.goalText = trimmed;
		goal.active = true;
		goal.iteration = 0;
		persist();

		try {
			pi.setSessionName(`🎯 ${trimmed.slice(0, 60)}`);
		} catch {
			// setSessionName may fail in non-interactive modes; ignore.
		}
		updateStatus(ctx, "active");
		notify(ctx, `Goal set: ${trimmed}`, "info");

		// active is true, so the upcoming agent_end will run the evaluator.
		pi.sendUserMessage(formatKickoff(trimmed, goal.config.maxIterations));
	}

	function continueGoal(ctx: ExtensionContext): void {
		if (!goal.goalText) {
			notify(ctx, "No goal set. Use /goal <description> to set one.", "warning");
			return;
		}
		if (goal.active) {
			notify(ctx, "Goal is already active.", "info");
			return;
		}
		if (!ctx.isIdle()) {
			notify(ctx, "Agent is busy. Press Escape to abort, then run /goal continue.", "warning");
			return;
		}

		bumpEpoch();
		stopRequested = false;
		evaluatorAbort?.abort();
		evaluatorAbort = null;
		consecutiveErrors = 0;
		goal.active = true;
		goal.iteration = 0; // fresh iteration budget for this resume
		persist();
		updateStatus(ctx, "active");
		notify(ctx, "Goal resumed.", "info");
		pi.sendUserMessage(formatResume(goal.goalText, goal.iteration, goal.config.maxIterations));
	}

	function stopGoal(ctx: ExtensionContext): void {
		if (!goal.goalText && !goal.active) {
			notify(ctx, "No goal is set.", "info");
			return;
		}
		bumpEpoch();
		stopRequested = true;
		evaluatorAbort?.abort();
		evaluatorAbort = null;
		const wasActive = goal.active;
		goal.active = false;
		consecutiveErrors = 0;
		persist();
		if (wasActive && !ctx.isIdle()) {
			ctx.abort(); // stop the in-flight turn; its agent_end will see active=false
		}
		updateStatus(ctx, "paused");
		pushGoalMessage("⏹ Goal stopped. Use /goal continue to resume or /goal clear to remove it.");
		notify(ctx, "Goal stopped.", "info");
	}

	function clearGoal(ctx: ExtensionContext): void {
		bumpEpoch();
		stopRequested = true;
		evaluatorAbort?.abort();
		evaluatorAbort = null;
		goal = defaultState();
		persist();
		ctx.ui.setStatus("goal", undefined);
		ctx.ui.setWidget("goal", undefined);
		notify(ctx, "Goal cleared.", "info");
	}

	function showStatus(ctx: ExtensionContext): void {
		if (!goal.goalText) {
			notify(ctx, "No goal set. Use /goal <description> to set one.", "info");
			return;
		}
		const state = goal.active ? (evaluating ? "evaluating" : "active") : "paused";
		const lines = [
			`Goal: ${goal.goalText}`,
			`Status: ${state}`,
			`Iteration: ${goal.iteration}/${goal.config.maxIterations}`,
			`Evaluator model: ${goal.config.evaluatorModel ?? "default"}`,
			`Evaluator tools: ${goal.config.evaluatorTools.join(", ")}`,
		];
		notify(ctx, lines.join("\n"), "info");
	}

	function showHelp(ctx: ExtensionContext): void {
		const help = [
			"/goal <description>            Set a goal and start the evaluator loop",
			"/goal set <text>              Set a goal (explicit form)",
			"/goal status                  Show current goal, iteration, and config",
			"/goal continue                Resume a paused goal",
			"/goal stop                    Stop the evaluator loop (keeps the goal paused)",
			"/goal clear                   Clear the goal entirely",
			"/goal config                  Show evaluator config",
			"/goal config max-iterations <n>           Set the iteration cap (default 10)",
			"/goal config evaluator-model <pattern>    Set the evaluator model (or 'default')",
			"/goal config evaluator-tools <a,b,…>       Set the evaluator's tools",
			"/goal help                    Show this help",
		];
		notify(ctx, help.join("\n"), "info");
	}

	function handleConfig(ctx: ExtensionContext, args: string): void {
		const parts = args.trim().split(/\s+/).filter(Boolean);
		if (parts.length === 0) {
			notify(
				ctx,
				[
					"Evaluator config:",
					`  max-iterations: ${goal.config.maxIterations}`,
					`  evaluator-model: ${goal.config.evaluatorModel ?? "default"}`,
					`  evaluator-tools: ${goal.config.evaluatorTools.join(", ")}`,
					"",
					"Change with: /goal config <key> <value>",
				].join("\n"),
				"info",
			);
			return;
		}
		const key = parts[0];
		const value = parts.slice(1).join(" ");
		if (key === "max-iterations") {
			const n = Number.parseInt(value, 10);
			if (!Number.isFinite(n) || n < 1) {
				notify(ctx, "max-iterations must be a positive integer.", "error");
				return;
			}
			goal.config.maxIterations = n;
			persist();
			updateStatus(ctx);
			notify(ctx, `max-iterations set to ${n}`, "info");
		} else if (key === "evaluator-model") {
			goal.config.evaluatorModel = value.toLowerCase() === "default" || value === "" ? undefined : value;
			persist();
			notify(ctx, `evaluator-model set to ${goal.config.evaluatorModel ?? "default"}`, "info");
		} else if (key === "evaluator-tools") {
			const tools = value
				.split(",")
				.map((t) => t.trim())
				.filter(Boolean);
			if (tools.length === 0) {
				notify(ctx, "evaluator-tools requires a comma-separated list (e.g. read,grep,find,ls,bash).", "error");
				return;
			}
			goal.config.evaluatorTools = tools;
			persist();
			notify(ctx, `evaluator-tools set to ${tools.join(", ")}`, "info");
		} else {
			notify(ctx, `Unknown config key: ${key}. Use max-iterations, evaluator-model, or evaluator-tools.`, "error");
		}
	}

	/* ------------------------------------------------------------------ */
	/* /goal command                                                       */
	/* ------------------------------------------------------------------ */

	pi.registerCommand("goal", {
		description: "Set a goal and run an evaluator loop, or manage the active goal",
		handler: async (args, ctx) => {
			const trimmed = (args ?? "").trim();

			if (trimmed === "" || trimmed === "help") {
				if (!goal.goalText) showHelp(ctx);
				else showStatus(ctx);
				return;
			}
			if (trimmed === "status") return showStatus(ctx);
			if (trimmed === "stop") return stopGoal(ctx);
			if (trimmed === "clear") return clearGoal(ctx);
			if (trimmed === "continue") return continueGoal(ctx);
			if (trimmed === "config") return handleConfig(ctx, "");

			if (trimmed === "set" || trimmed.startsWith("set ")) {
				await startGoal(ctx, trimmed.slice(3).trim());
				return;
			}
			if (trimmed.startsWith("config ")) {
				return handleConfig(ctx, trimmed.slice(7));
			}

			// Implicit form: treat the whole argument as the goal text.
			await startGoal(ctx, trimmed);
		},
	});

	/* ------------------------------------------------------------------ */
	/* goal_status tool (lets the agent recall the active goal)            */
	/* ------------------------------------------------------------------ */

	pi.registerTool({
		name: "goal_status",
		label: "Goal Status",
		description:
			"Get the current goal, iteration count, and max iterations for the /goal evaluator loop. Use when you need to recall the active goal while working autonomously.",
		promptSnippet: "Check the current /goal evaluator-loop goal and iteration count",
		promptGuidelines: [
			"Use goal_status to recall the active goal and current iteration when working under a /goal evaluator loop.",
		],
		parameters: Type.Object({}),
		async execute() {
			const text = formatGoalForAgent(goal);
			return {
				content: [{ type: "text", text }],
				details: {
					goal: goal.goalText,
					active: goal.active,
					iteration: goal.iteration,
					maxIterations: goal.config.maxIterations,
				},
			};
		},
		renderCall(_args, theme) {
			return new Text(theme.fg("accent", theme.bold("goal_status")), 0, 0);
		},
		renderResult(result, _options, theme) {
			const text = result.content[0]?.type === "text" ? result.content[0].text : "";
			return new Text(theme.fg("muted", text), 0, 0);
		},
	});

	/* ------------------------------------------------------------------ */
	/* Message renderer for goal notifications                              */
	/* ------------------------------------------------------------------ */

	pi.registerMessageRenderer(GOAL_MESSAGE_TYPE, (message, _options, theme) => {
		const content =
			typeof message.content === "string"
				? message.content
				: Array.isArray(message.content)
					? message.content
							.filter((c): c is { type: "text"; text: string } => typeof c === "object" && c.type === "text")
							.map((c) => c.text)
							.join("\n")
					: "";
		return new Text(theme.fg("accent", content), 0, 0);
	});

	/* ------------------------------------------------------------------ */
	/* The evaluator loop                                                  */
	/* ------------------------------------------------------------------ */

	pi.on("agent_end", async (event: AgentEndEvent, ctx: ExtensionContext) => {
		if (!goal.active || !goal.goalText) return;
		if (evaluating || stopRequested) return;

		// If the agent was aborted or errored, pause the loop and keep the goal.
		const lastAssistant = findLastAssistant(event.messages);
		if (lastAssistant && (lastAssistant.stopReason === "aborted" || lastAssistant.stopReason === "error")) {
			pauseGoal(ctx, `agent ${lastAssistant.stopReason}`);
			return;
		}

		if (goal.iteration >= goal.config.maxIterations) {
			pauseGoal(ctx, `reached max iterations (${goal.config.maxIterations})`);
			notify(
				ctx,
				`🎯 Reached the iteration cap (${goal.config.maxIterations}). Use /goal continue to resume with a fresh budget, or /goal clear to stop.`,
				"warning",
			);
			return;
		}

		const epoch = goalEpoch;
		evaluating = true;
		evaluatorAbort = new AbortController();
		updateStatus(ctx, "evaluating");

		let result: EvaluationResult | null = null;
		try {
			const transcript = buildTranscript(ctx.sessionManager.getBranch());
			result = await runEvaluator({
				goal: goal.goalText,
				transcript,
				config: goal.config,
				cwd: ctx.cwd,
				signal: evaluatorAbort.signal,
				onProgress: (label) => updateStatus(ctx, "evaluating", label),
			});
		} catch (err) {
			// runEvaluator throws on abort or unexpected failure. Only treat as a
			// real error if the goal is still the active one we started with.
			if (!stopRequested && epoch === goalEpoch) {
				result = {
					achieved: false,
					confidence: "low",
					summary: "",
					gaps: [],
					nextAction: "",
					raw: "",
					error: err instanceof Error ? err.message : String(err),
				};
			}
		}

		evaluating = false;
		evaluatorAbort = null;

		// A /goal command may have changed or stopped the goal during evaluation.
		if (stopRequested || epoch !== goalEpoch || !result) {
			if (goal.active && goal.goalText) updateStatus(ctx, "active");
			return;
		}

		goal.iteration += 1;
		persist();

		if (result.error) {
			consecutiveErrors += 1;
			if (consecutiveErrors >= 2) {
				pauseGoal(ctx, `evaluator failed repeatedly (${result.error})`);
				return;
			}
			injectFeedback(ctx, result, true);
		} else {
			consecutiveErrors = 0;
			if (result.achieved) {
				completeGoal(ctx, result);
				return;
			}
			injectFeedback(ctx, result, false);
		}

		if (goal.active && goal.goalText) updateStatus(ctx, "active");
	});

	/* ------------------------------------------------------------------ */
	/* State persistence and cleanup                                       */
	/* ------------------------------------------------------------------ */

	pi.on("session_start", async (_event, ctx: ExtensionContext) => {
		const entries = ctx.sessionManager.getEntries();
		goal = restoreGoalFromEntries(entries);
		evaluating = false;
		stopRequested = false;
		consecutiveErrors = 0;
		evaluatorAbort = null;
		bumpEpoch();
		updateStatus(ctx, goal.active ? "active" : "paused");
	});

	pi.on("session_shutdown", async () => {
		stopRequested = true;
		evaluatorAbort?.abort();
		evaluatorAbort = null;
	});
}
