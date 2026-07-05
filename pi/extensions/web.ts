/**
 * Web Extension for pi
 *
 * Gives the agent two tools for accessing the web:
 *   - web_fetch  : fetch an http(s) URL and return cleaned-up text/markdown
 *                  (HTML is converted to readable markdown or plain text;
 *                   JSON is pretty-printed; a simple CSS-ish selector can
 *                   extract just part of a page)
 *   - web_search : search the web via DuckDuckGo (no API key) and return
 *                  result titles, URLs, and snippets
 *
 * No external dependencies — uses Node's built-in global `fetch` (Node >= 18)
 * and `typebox` for the parameter schema. HTML parsing is intentionally
 * lightweight (regex-based) and best-effort; it is meant for reading content,
 * not for fidelity-perfect conversion.
 *
 * Install: this file lives at ~/.pi/agent/extensions/web.ts and is
 * auto-discovered globally. Reload with `/reload` or restart pi to activate.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const USER_AGENT =
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 (compatible; pi-web-tool)";

const DEFAULT_MAX_LENGTH = 20000; // cap on returned text (chars)
const MIN_MAX_LENGTH = 100;
const MAX_MAX_LENGTH = 200000;
const FETCH_TIMEOUT_MS = 20000; // per-request timeout for web_fetch
const SEARCH_TIMEOUT_MS = 20000; // per-request timeout for web_search
const MAX_BYTES = 3_000_000; // hard cap on downloaded body (bytes)
const DEFAULT_SEARCH_RESULTS = 8;
const MAX_SEARCH_RESULTS = 20;

const TEXTLIKE = /text\/|application\/(json|xml|javascript|xhtml|x-www-form-urlencoded)|\+json|\+xml/i;
const HTMLLIKE = /text\/html|application\/xhtml/i;

// ---------------------------------------------------------------------------
// AbortSignal helper: combine a parent signal with a timeout
// ---------------------------------------------------------------------------

function withTimeout(parent: AbortSignal | undefined, ms: number): AbortSignal {
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(new Error(`timeout after ${ms}ms`)), ms);
	if (parent) {
		if (parent.aborted) {
			ctrl.abort(parent.reason);
		} else {
			parent.addEventListener("abort", () => ctrl.abort(parent.reason), { once: true });
		}
	}
	ctrl.signal.addEventListener("abort", () => clearTimeout(timer), { once: true });
	return ctrl.signal;
}

// ---------------------------------------------------------------------------
// HTML entity decoding
// ---------------------------------------------------------------------------

const NAMED_ENTITIES: Record<string, string> = {
	amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: "\u00a0",
	copy: "\u00a9", reg: "\u00ae", trade: "\u2122", hellip: "\u2026",
	mdash: "\u2014", ndash: "\u2013", lsquo: "\u2018", rsquo: "\u2019",
	ldquo: "\u201c", rdquo: "\u201d", laquo: "\u00ab", raquo: "\u00bb",
	deg: "\u00b0", plusmn: "\u00b1", times: "\u00d7", divide: "\u00f7",
	euro: "\u20ac", pound: "\u00a3", cent: "\u00a2", yen: "\u00a5",
	sect: "\u00a7", para: "\u00b6", middot: "\u00b7", bull: "\u2022",
	dagger: "\u2020", Dagger: "\u2021", permil: "\u2030", prime: "\u2032",
	Prime: "\u2033", infin: "\u221e", ne: "\u2260", le: "\u2264", ge: "\u2265",
	larr: "\u2190", uarr: "\u2191", rarr: "\u2192", darr: "\u2193",
	harr: "\u2194", lArr: "\u21d0", rArr: "\u21d2", forall: "\u2200",
	exist: "\u2203", empty: "\u2205", isin: "\u2208", notin: "\u2209",
	sum: "\u2211", prod: "\u220f", minus: "\u2212", radic: "\u221a",
	prop: "\u221d", ang: "\u2220", cap: "\u2229", cup: "\u222a",
	int: "\u222b", there4: "\u2234", sim: "\u223c", cong: "\u2245",
	asymp: "\u2248", sub: "\u2282", sup: "\u2283", nsub: "\u2284",
	sube: "\u2286", supe: "\u2287", oplus: "\u2295", otimes: "\u2297",
	perp: "\u22a5", sdot: "\u22c5", alpha: "\u03b1", beta: "\u03b2",
	gamma: "\u03b3", delta: "\u03b4", epsilon: "\u03b5", zeta: "\u03b6",
	eta: "\u03b7", theta: "\u03b8", iota: "\u03b9", kappa: "\u03ba",
	lambda: "\u03bb", mu: "\u03bc", nu: "\u03bd", xi: "\u03be",
	omicron: "\u03bf", pi: "\u03c0", rho: "\u03c1", sigma: "\u03c3",
	tau: "\u03c4", upsilon: "\u03c5", phi: "\u03c6", chi: "\u03c7",
	psi: "\u03c8", omega: "\u03c9", Alpha: "\u0391", Beta: "\u0392",
	Gamma: "\u0393", Delta: "\u0394", Epsilon: "\u0395", Zeta: "\u0396",
	Eta: "\u0397", Theta: "\u0398", Iota: "\u0399", Kappa: "\u039a",
	Lambda: "\u039b", Mu: "\u039c", Nu: "\u039d", Xi: "\u039e",
	Omicron: "\u039f", Pi: "\u03a0", Rho: "\u03a1", Sigma: "\u03a3",
	Tau: "\u03a4", Upsilon: "\u03a5", Phi: "\u03a6", Chi: "\u03a7",
	Psi: "\u03a8", Omega: "\u03a9",
};

function decodeEntities(s: string): string {
	return s.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/g, (m, ent: string) => {
		if (ent.startsWith("#")) {
			const code = ent[1] === "x" || ent[1] === "X"
				? parseInt(ent.slice(2), 16)
				: parseInt(ent.slice(1), 10);
			return Number.isNaN(code) ? m : String.fromCodePoint(code);
		}
		return Object.prototype.hasOwnProperty.call(NAMED_ENTITIES, ent)
			? NAMED_ENTITIES[ent]
			: m;
	});
}

// ---------------------------------------------------------------------------
// HTML utilities
// ---------------------------------------------------------------------------

function removeBlocks(html: string, tag: string): string {
	const re = new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?</${tag}\\s*>`, "gi");
	return html.replace(re, " ");
}

function extractTitle(html: string): string | null {
	const m = html.match(/<title[^>]*>([\s\S]*?)<\/title\s*>/i);
	return m ? decodeEntities(m[1]).replace(/\s+/g, " ").trim() : null;
}

/** Strip all tags and collapse whitespace to a single space. */
function stripTagsCompact(s: string): string {
	return decodeEntities(s.replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim();
}

function collapseWhitespace(text: string): string {
	return text
		.replace(/[ \t\f\v]+/g, " ")
		.replace(/ *\n */g, "\n")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

/**
 * Extract the inner HTML of the first element matching a simple selector.
 * Supported selector forms: "tag", ".class", "#id", "tag.class", "tag#id".
 * Returns null if nothing matches.
 */
function extractBySelector(html: string, selector: string): string | null {
	const idMatch = selector.match(/#([\w-]+)/);
	const clsMatch = selector.match(/\.([\w-]+)/);
	const tagMatch = selector.match(/^([a-zA-Z][\w-]*)/);
	const tagName = (tagMatch ? tagMatch[1] : "div").toLowerCase();
	const id = idMatch ? idMatch[1] : null;
	const cls = clsMatch ? clsMatch[1] : null;

	const openRe = new RegExp(`<${tagName}(\\s[^>]*|)>`, "gi");
	let m: RegExpExecArray | null;
	while ((m = openRe.exec(html))) {
		const attrs = m[1] || "";
		if (id && !new RegExp(`\\sid\\s*=\\s*["']?${id}["']?`, "i").test(attrs)) continue;
		if (cls && !new RegExp(`\\sclass\\s*=\\s*["'][^"']*\\b${cls}\\b`, "i").test(attrs)) continue;

		const start = m.index + m[0].length;
		const openTagRe = new RegExp(`<${tagName}\\b`, "gi");
		const closeTagRe = new RegExp(`</${tagName}\\s*>`, "gi");
		let depth = 1;
		let pos = start;
		let guard = 0;
		while (depth > 0 && guard++ < 200000) {
			openTagRe.lastIndex = pos;
			closeTagRe.lastIndex = pos;
			const o = openTagRe.exec(html);
			const c = closeTagRe.exec(html);
			if (!c) return null;
			if (o && o.index < c.index) {
				depth++;
				pos = o.index + o[0].length;
			} else {
				depth--;
				pos = c.index + c[0].length;
				if (depth === 0) return html.slice(start, c.index);
			}
		}
	}
	return null;
}

function htmlToText(html: string): string {
	let s = html;
	s = removeBlocks(s, "script");
	s = removeBlocks(s, "style");
	s = removeBlocks(s, "noscript");
	s = removeBlocks(s, "template");
	s = removeBlocks(s, "svg");
	s = removeBlocks(s, "head");
	s = s.replace(/<!--[\s\S]*?-->/g, " ");
	s = s.replace(/<br\s*\/?>/gi, "\n");
	s = s.replace(/<hr\s*\/?>/gi, "\n---\n");
	s = s.replace(/<li\b[^>]*>/gi, "\n- ");
	s = s.replace(/<\/(p|div|section|article|header|footer|nav|aside|main|li|tr|table|thead|tbody|tfoot|blockquote|figure|figcaption|address|fieldset|form|dl|dd|dt|h[1-6])\s*>/gi, "\n");
	s = s.replace(/<(p|div|section|article|header|footer|nav|aside|main|blockquote|figure|figcaption|address|fieldset|form|dl|dd|dt|h[1-6])\b[^>]*>/gi, "\n");
	s = s.replace(/<[^>]+>/g, "");
	s = decodeEntities(s);
	return collapseWhitespace(s);
}

// Sentinels (private-use area) to protect pre/code content from later passes.
const PRE_PREFIX = "\uE000P";
const CODE_PREFIX = "\uE000C";
const SENTINEL_RE = /\uE000([PC])(\d+)\uE001/g;

function htmlToMarkdown(html: string): string {
	let s = html;
	s = removeBlocks(s, "script");
	s = removeBlocks(s, "style");
	s = removeBlocks(s, "noscript");
	s = removeBlocks(s, "template");
	s = removeBlocks(s, "head");
	s = s.replace(/<!--[\s\S]*?-->/g, " ");

	// Protect <pre> and inline <code> first so later tag-stripping leaves them.
	const codeBlocks: string[] = [];
	s = s.replace(/<pre\b[^>]*>([\s\S]*?)<\/pre\s*>/gi, (_m, inner) => {
		const code = decodeEntities(inner.replace(/<[^>]+>/g, "")).replace(/^\n+|\n+$/g, "");
		codeBlocks.push(code);
		return `\n${PRE_PREFIX}${codeBlocks.length - 1}\uE001\n`;
	});
	s = s.replace(/<code\b[^>]*>([\s\S]*?)<\/code\s*>/gi, (_m, inner) => {
		const code = decodeEntities(inner);
		codeBlocks.push(code);
		return `${CODE_PREFIX}${codeBlocks.length - 1}\uE001`;
	});

	// Images (alt-first, then src-only).
	s = s.replace(/<img\b[^>]*\balt=["']([^"']*)["'][^>]*>/gi, (_m, alt) => `![${alt}]`);
	s = s.replace(/<img\b[^>]*\bsrc=["']([^"']*)["'][^>]*>/gi, (_m, src) => `[image](${src})`);

	// Links.
	s = s.replace(/<a\b([^>]*)>([\s\S]*?)<\/a\s*>/gi, (_m, attrs, inner) => {
		const hrefMatch = attrs.match(/href\s*=\s*["']([^"']*)["']/i);
		const href = hrefMatch ? hrefMatch[1].trim() : "";
		const text = stripTagsCompact(inner) || href;
		if (href && text) return `[${text}](${href})`;
		return text;
	});

	// Emphasis.
	s = s.replace(/<(strong|b)\b[^>]*>([\s\S]*?)<\/\1\s*>/gi, (_m, _t, inner) => `**${inner.trim()}**`);
	s = s.replace(/<(em|i)\b[^>]*>([\s\S]*?)<\/\1\s*>/gi, (_m, _t, inner) => `*${inner.trim()}*`);

	// Headings.
	s = s.replace(/<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1\s*>/gi, (_m, level, inner) => {
		const text = stripTagsCompact(inner);
		return `\n\n${"#".repeat(Number(level))} ${text}\n\n`;
	});

	// Line breaks / rules.
	s = s.replace(/<br\s*\/?>/gi, "\n");
	s = s.replace(/<hr\s*\/?>/gi, "\n---\n");

	// List items.
	s = s.replace(/<li\b[^>]*>([\s\S]*?)<\/li\s*>/gi, (_m, inner) => `\n- ${stripTagsCompact(inner)}`);

	// Blockquotes.
	s = s.replace(/<blockquote\b[^>]*>([\s\S]*?)<\/blockquote\s*>/gi, (_m, inner) => {
		const lines = stripTagsCompact(inner).split("\n");
		return "\n" + lines.map((l) => `> ${l}`).join("\n") + "\n";
	});

	// Block-level open/close → newlines.
	s = s.replace(/<\/?(p|div|section|article|header|footer|nav|aside|main|ul|ol|table|thead|tbody|tfoot|tr|td|th|figure|figcaption|address|fieldset|form|dl|dd|dt)\b[^>]*>/gi, "\n");

	// Strip remaining tags.
	s = s.replace(/<[^>]+>/g, "");

	s = decodeEntities(s);
	s = collapseWhitespace(s);

	// Restore protected code blocks.
	s = s.replace(SENTINEL_RE, (_m, kind, i) => {
		const code = codeBlocks[Number(i)] ?? "";
		return kind === "P" ? `\n\`\`\`\n${code}\n\`\`\`\n` : `\`${code}\``;
	});

	return collapseWhitespace(s);
}

// ---------------------------------------------------------------------------
// Response reading (capped)
// ---------------------------------------------------------------------------

async function readCapped(
	res: Response,
	maxBytes: number,
): Promise<{ text: string; truncated: boolean; bytes: number }> {
	if (!res.body) {
		const t = await res.text();
		return { text: t, truncated: t.length > maxBytes, bytes: t.length };
	}
	const reader = res.body.getReader();
	const decoder = new TextDecoder("utf-8", { fatal: false });
	let out = "";
	let bytes = 0;
	let truncated = false;
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			bytes += value.length;
			out += decoder.decode(value, { stream: true });
			if (bytes >= maxBytes) {
				truncated = true;
				break;
			}
		}
		out += decoder.decode();
	} finally {
		try {
			reader.releaseLock();
		} catch {
			/* ignore */
		}
	}
	return { text: out, truncated, bytes };
}

// ---------------------------------------------------------------------------
// DuckDuckGo search parsing
// ---------------------------------------------------------------------------

function decodeDdgHref(href: string): string {
	let h = href.trim();
	if (h.startsWith("//")) h = "https:" + h;
	try {
		const u = new URL(h);
		const uddg = u.searchParams.get("uddg");
		if (uddg) return decodeURIComponent(uddg);
		return u.toString();
	} catch {
		return h;
	}
}

function parseDdg(html: string, max: number): Array<{ title: string; url: string; snippet: string }> {
	const out: Array<{ title: string; url: string; snippet: string }> = [];
	const seen = new Set<string>();

	const snippets: string[] = [];
	const snippetRe = /<a\b[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a\s*>/gi;
	let sm: RegExpExecArray | null;
	while ((sm = snippetRe.exec(html))) snippets.push(stripTagsCompact(sm[1]));

	const linkRe = /<a\b[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a\s*>/gi;
	let lm: RegExpExecArray | null;
	let i = 0;
	while ((lm = linkRe.exec(html)) && out.length < max) {
		const url = decodeDdgHref(lm[1]);
		const title = stripTagsCompact(lm[2]);
		if (!url || !title) {
			i++;
			continue;
		}
		if (seen.has(url)) {
			i++;
			continue;
		}
		seen.add(url);
		out.push({ title, url, snippet: snippets[i] || "" });
		i++;
	}
	return out;
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function webExtension(pi: ExtensionAPI) {
	// -----------------------------------------------------------------------
	// web_fetch
	// -----------------------------------------------------------------------
	pi.registerTool({
		name: "web_fetch",
		label: "Web Fetch",
		description:
			"Fetch an http(s) URL and return its content as readable text. HTML pages are converted to markdown (default) or plain text; JSON is pretty-printed; other text is returned raw. Use this to read a specific page or API endpoint. Use format 'raw' to skip HTML conversion, and 'selector' (e.g. 'article', 'main', '.content', '#body') to extract just part of an HTML page.",
		promptSnippet: "Fetch a URL and return its content as markdown/text",
		promptGuidelines: [
			"Use web_fetch to retrieve and read the content of a specific http or https URL. It returns cleaned, readable text (HTML→markdown by default), so prefer it over shelling out to curl for reading pages.",
			"Use web_search first when you need to find relevant pages, then web_fetch a result URL to read the full content. For JSON APIs, web_fetch pretty-prints the response automatically.",
		],
		parameters: Type.Object({
			url: Type.String({ description: "Absolute http(s) URL to fetch" }),
			format: Type.Optional(
				Type.Union(
					[Type.Literal("markdown"), Type.Literal("text"), Type.Literal("raw")],
					{ description: "Output format: 'markdown' (default, best for articles), 'text' (plain text), or 'raw' (no HTML conversion)" },
				),
			),
			maxLength: Type.Optional(
				Type.Number({
					description: `Maximum characters of converted text to return (default ${DEFAULT_MAX_LENGTH}, max ${MAX_MAX_LENGTH}). Output is truncated if longer.`,
				}),
			),
			selector: Type.Optional(
				Type.String({
					description: "Optional simple CSS-ish selector to extract part of an HTML page before conversion: 'tag', '.class', '#id', 'tag.class', or 'tag#id' (e.g. 'article', 'main', '.content', '#body')",
				}),
			),
		}),

		async execute(_toolCallId, params: any, signal: AbortSignal | undefined, onUpdate?: (u: any) => void) {
			const url = String(params?.url ?? "").trim();
			if (!/^https?:\/\//i.test(url)) {
				return {
					content: [{ type: "text" as const, text: `Error: URL must start with http:// or https:// (got: ${url || "(empty)"})` }],
					details: { url, error: "invalid url" },
					isError: true,
				};
			}

			const format = (params?.format === "text" || params?.format === "raw") ? params.format : "markdown";
			const maxLength = Math.min(
				Math.max(Number(params?.maxLength) || DEFAULT_MAX_LENGTH, MIN_MAX_LENGTH),
				MAX_MAX_LENGTH,
			);
			const selector = params?.selector ? String(params.selector) : null;

			onUpdate?.({ content: [{ type: "text", text: `Fetching ${url} …` }] });

			let res: Response;
			try {
				res = await fetch(url, {
					headers: {
						"User-Agent": USER_AGENT,
						Accept: "text/html,application/xhtml+xml,application/json,text/plain,application/xml,*/*;q=0.8",
						"Accept-Language": "en-US,en;q=0.9",
					},
					redirect: "follow",
					signal: withTimeout(signal, FETCH_TIMEOUT_MS),
				});
			} catch (e: any) {
				const aborted = e?.name === "AbortError" || /timeout/i.test(String(e?.message ?? ""));
				const msg = aborted
					? `Request timed out or was cancelled (limit ${FETCH_TIMEOUT_MS}ms)`
					: `Fetch failed: ${e?.message ?? String(e)}`;
				return {
					content: [{ type: "text" as const, text: `Error: ${msg}` }],
					details: { url, error: msg },
					isError: true,
				};
			}

			const contentType = res.headers.get("content-type") || "";
			const finalUrl = res.url || url;
			const status = res.status;

			const { text: raw, truncated: byteTruncated, bytes } = await readCapped(res, MAX_BYTES);
			const isHtml = HTMLLIKE.test(contentType);
			const textLike = TEXTLIKE.test(contentType);

			let body: string;
			let usedFormat: string = format;

			if (!textLike && format !== "raw") {
				body = `Binary or non-text content (Content-Type: ${contentType || "unknown"}). ${bytes} bytes downloaded.${byteTruncated ? " Download was capped." : ""} Use format "raw" to retrieve the raw bytes as text (may be garbled for binary).`;
				usedFormat = "binary";
			} else if (format === "raw") {
				body = raw;
				usedFormat = "raw";
			} else if (isHtml) {
				let html = raw;
				let selectorMiss = false;
				if (selector) {
					const frag = extractBySelector(raw, selector);
					if (frag != null) {
						html = frag;
					} else {
						selectorMiss = true;
					}
				}
				const converted = format === "text" ? htmlToText(html) : htmlToMarkdown(html);
				const title = extractTitle(raw);
				const headerParts: string[] = [];
				if (title) headerParts.push(`# ${title}`);
				if (selectorMiss) headerParts.push(`<!-- selector "${selector}" matched nothing; showing full page -->`);
				body = headerParts.length ? `${headerParts.join("\n")}\n\n${converted}` : converted;
			} else if (/application\/json/i.test(contentType)) {
				try {
					body = JSON.stringify(JSON.parse(raw), null, 2);
					usedFormat = "json";
				} catch {
					body = raw;
					usedFormat = "raw";
				}
			} else {
				body = raw;
				usedFormat = "raw";
			}

			let outputTruncated = false;
			if (body.length > maxLength) {
				body = body.slice(0, maxLength);
				outputTruncated = true;
			}

			const meta = [
				`URL: ${finalUrl}`,
				`Status: ${status}`,
				`Content-Type: ${contentType || "(none)"}`,
				`Format: ${usedFormat}${selector ? ` (selector: ${selector})` : ""}`,
				`Bytes: ${bytes.toLocaleString()}${byteTruncated ? " (download capped at " + MAX_BYTES.toLocaleString() + ")" : ""}`,
				outputTruncated ? `Truncated: output limited to ${maxLength.toLocaleString()} chars` : null,
			]
				.filter((x): x is string => x !== null)
				.join("\n");

			const text = `${meta}\n\n${body}`;

			return {
				content: [{ type: "text" as const, text }],
				details: {
					url: finalUrl,
					requestedUrl: url,
					status,
					contentType,
					format: usedFormat,
					selector: selector ?? undefined,
					bytes,
					byteTruncated,
					outputTruncated,
					maxLength,
				},
				isError: status >= 400,
			};
		},
	});

	// -----------------------------------------------------------------------
	// web_search
	// -----------------------------------------------------------------------
	pi.registerTool({
		name: "web_search",
		label: "Web Search",
		description:
			"Search the web via DuckDuckGo (no API key required) and return a list of results with titles, URLs, and snippets. Use this to find current information or locate relevant pages; then use web_fetch to read a result's full content.",
		promptSnippet: "Search the web (DuckDuckGo) and return result titles, URLs, snippets",
		promptGuidelines: [
			"Use web_search when you need up-to-date information or need to discover relevant pages. Then use web_fetch on a promising result URL to read the full content.",
			"web_search returns titles, URLs, and short snippets only — it does not fetch page contents. Always follow up with web_fetch for detail.",
		],
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			maxResults: Type.Optional(
				Type.Number({
					description: `Maximum number of results to return (default ${DEFAULT_SEARCH_RESULTS}, max ${MAX_SEARCH_RESULTS})`,
				}),
			),
		}),

		async execute(_toolCallId, params: any, signal: AbortSignal | undefined, onUpdate?: (u: any) => void) {
			const query = String(params?.query ?? "").trim();
			if (!query) {
				return {
					content: [{ type: "text" as const, text: "Error: query is required" }],
					details: { error: "missing query" },
					isError: true,
				};
			}
			const max = Math.min(Math.max(Number(params?.maxResults) || DEFAULT_SEARCH_RESULTS, 1), MAX_SEARCH_RESULTS);

			onUpdate?.({ content: [{ type: "text", text: `Searching the web for: ${query} …` }] });

			const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}&kp=-2`;

			let res: Response;
			try {
				res = await fetch(url, {
					headers: {
						"User-Agent": USER_AGENT,
						Accept: "text/html,application/xhtml+xml",
						"Accept-Language": "en-US,en;q=0.9",
					},
					redirect: "follow",
					signal: withTimeout(signal, SEARCH_TIMEOUT_MS),
				});
			} catch (e: any) {
				const aborted = e?.name === "AbortError" || /timeout/i.test(String(e?.message ?? ""));
				const msg = aborted
					? `Search timed out (limit ${SEARCH_TIMEOUT_MS}ms)`
					: `Search failed: ${e?.message ?? String(e)}`;
				return {
					content: [{ type: "text" as const, text: `Error: ${msg}` }],
					details: { query, error: msg },
					isError: true,
				};
			}

			if (!res.ok) {
				return {
					content: [{ type: "text" as const, text: `Search request failed (HTTP ${res.status} ${res.statusText}).` }],
					details: { query, status: res.status },
					isError: true,
				};
			}

			const html = await res.text();
			const results = parseDdg(html, max);

			if (results.length === 0) {
				const blocked = /anomaly|captcha|are you a robot|blocked/i.test(html);
				const note = blocked
					? `No results found for "${query}". DuckDuckGo may have returned a verification/block page; try again later or use web_fetch on a specific URL.`
					: `No results found for "${query}".`;
				return {
					content: [{ type: "text" as const, text: note }],
					details: { query, results: [], status: res.status, blocked },
				};
			}

			const lines = [`Search: ${query}`, `Results: ${results.length}`, ""];
			results.forEach((r, i) => {
				lines.push(`${i + 1}. ${r.title}`);
				lines.push(`   ${r.url}`);
				if (r.snippet) lines.push(`   ${r.snippet}`);
				lines.push("");
			});

			return {
				content: [{ type: "text" as const, text: lines.join("\n").trim() }],
				details: { query, results, status: res.status },
			};
		},
	});
}
