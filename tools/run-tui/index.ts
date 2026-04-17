#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";

type JsonObject = Record<string, unknown>;

type ReviewDecision = "accept" | "rewrite" | "drop" | "unknown";
type ReviewStage = "prepare_only" | "generated" | "reviewed" | "finalized";
type ChangeKind =
  | "accepted_unchanged"
  | "rewritten_changed"
  | "rewritten_unchanged"
  | "dropped"
  | "pending";
type FilterKey =
  | "all"
  | "accept"
  | "rewrite"
  | "drop"
  | "rewritten_unchanged"
  | "flagged"
  | "reviewed"
  | "generated"
  | "requested"
  | "source_only";
type DetailTab = "overview" | "generator" | "reviewer" | "compare" | "raw";

type RunArtifacts = {
  manifest?: JsonObject;
  sampleRows: JsonObject[];
  generationRequests: JsonObject[];
  generationOutputs: JsonObject[];
  reviewRequests: JsonObject[];
  reviewOutputs: JsonObject[];
  trainingRows: JsonObject[];
  droppedRows: JsonObject[];
};

type RunEntry = {
  index: number;
  paperId: string;
  title: string;
  abstract: string;
  primaryCategory: string;
  categories: string[];
  createdDate: string;
  reviewDecision: ReviewDecision;
  reviewStage: ReviewStage;
  changeKind: ChangeKind;
  candidateSummary: string;
  finalSummary: string;
  dropReason: string;
  candidateFlags: string[];
  finalFlags: string[];
  hasCandidate: boolean;
  hasFinal: boolean;
  sampleRow?: JsonObject;
  generationRequest?: JsonObject;
  generationOutput?: JsonObject;
  reviewRequest?: JsonObject;
  reviewOutput?: JsonObject;
  trainingRow?: JsonObject;
  droppedRow?: JsonObject;
  searchBlob: string;
};

type RunCounts = {
  total: number;
  decision: Record<ReviewDecision, number>;
  stage: {
    finalized: number;
    reviewed: number;
    generated: number;
    requested: number;
    source_only: number;
  };
  change: Record<ChangeKind, number>;
  flagged: number;
};

type RunData = {
  name: string;
  dir: string;
  manifest?: JsonObject;
  entries: RunEntry[];
  counts: RunCounts;
  files: string[];
};

type AppState = {
  runs: RunData[];
  rootDir: string;
  scopedRunDir?: string;
  runIndex: number;
  showRunInfo: boolean;
  filter: FilterKey;
  query: string;
  filteredEntries: RunEntry[];
  selectedIndex: number;
  listScroll: number;
  detailScroll: number;
  detailLineCount: number;
  detailViewportHeight: number;
  tab: DetailTab;
  mode: "browse" | "search";
  draftQuery: string;
  shouldExit: boolean;
};

const WORD_RE = /[A-Za-z0-9][A-Za-z0-9'/-]*/g;
const BANNED_OPENINGS = [
  "this paper",
  "we ",
  "the authors",
  "in this work",
  "in this paper",
] as const;

const FILTER_ORDER: FilterKey[] = [
  "all",
  "accept",
  "rewrite",
  "drop",
  "rewritten_unchanged",
  "flagged",
  "reviewed",
  "generated",
  "requested",
  "source_only",
];
const TAB_ORDER: DetailTab[] = ["overview", "compare", "generator", "reviewer", "raw"];
const ARTIFACT_FILE_ORDER = [
  "run_manifest.json",
  "sample_source_rows.jsonl",
  "generation_requests.jsonl",
  "generation_outputs.jsonl",
  "review_requests.jsonl",
  "review_outputs.jsonl",
  "training_rows.jsonl",
  "dropped_rows.jsonl",
] as const;

const ansi = {
  reset: "\u001b[0m",
  clear: "\u001b[2J\u001b[H",
  hideCursor: "\u001b[?25l",
  showCursor: "\u001b[?25h",
  altScreenOn: "\u001b[?1049h",
  altScreenOff: "\u001b[?1049l",
  dim: "\u001b[2m",
  bold: "\u001b[1m",
  cyan: "\u001b[36m",
  yellow: "\u001b[33m",
  green: "\u001b[32m",
  red: "\u001b[31m",
  magenta: "\u001b[35m",
  blue: "\u001b[34m",
  inverse: "\u001b[7m",
};

const detailCache = new Map<string, string[]>();

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const runs = loadRuns(args.runDir, args.rootDir);

  if (runs.length === 0) {
    console.error("No run directories with known artifacts were found.");
    process.exitCode = 1;
    return;
  }

  if (args.help) {
    printHelp(runs);
    return;
  }

  if (args.summary || !process.stdout.isTTY || !process.stdin.isTTY) {
    printSummary(runs);
    return;
  }

  const initialRunIndex = Math.max(0, runs.findIndex((run) => run.dir === args.runDir));
  const state: AppState = {
    runs,
    rootDir: args.rootDir,
    scopedRunDir: args.runDir,
    runIndex: initialRunIndex === -1 ? 0 : initialRunIndex,
    showRunInfo: false,
    filter: "all",
    query: "",
    filteredEntries: [],
    selectedIndex: 0,
    listScroll: 0,
    detailScroll: 0,
    detailLineCount: 0,
    detailViewportHeight: 0,
    tab: "overview",
    mode: "browse",
    draftQuery: "",
    shouldExit: false,
  };

  recomputeEntries(state);
  startInteractiveSession(state);
}

function parseArgs(argv: string[]): { help: boolean; summary: boolean; runDir?: string; rootDir: string } {
  let help = false;
  let summary = false;
  let runDir: string | undefined;
  let rootDir = path.resolve("hf_bucket/runs");

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (arg === "--summary") {
      summary = true;
      continue;
    }
    if (arg === "--root") {
      rootDir = path.resolve(argv[index + 1] ?? "hf_bucket/runs");
      index += 1;
      continue;
    }
    if (arg === "--run") {
      runDir = path.resolve(argv[index + 1] ?? "");
      index += 1;
      continue;
    }
    if (!arg.startsWith("-") && !runDir) {
      runDir = path.resolve(arg);
    }
  }

  return { help, summary, runDir, rootDir };
}

function printHelp(runs: RunData[]): void {
  const discovered = runs.map((run) => `  - ${run.name} (${run.entries.length} rows)`).join("\n");
  console.log(
    [
      "Prompt/run pair browser",
      "",
      "Usage:",
      "  npm run run-tui -- [run-dir]",
      "  npm run run-tui -- --run hf_bucket/runs/openrouter_smoke_100_v3",
      "  npm run run-tui -- --summary",
      "",
      "Keys:",
      "  Up/Down or j/k   move through rows",
      "  Left/Right or h/l cycle tabs",
      "  PageUp/PageDown  scroll the detail pane",
      "  Space/b or Ctrl-d/Ctrl-u scroll detail faster",
      "  1/2/3/4/5        jump to overview, compare, generator, reviewer, raw",
      "  i                toggle expanded run info",
      "  f                cycle filter",
      "  /                search titles, ids, categories, candidate/final text, decisions",
      "  c                clear search",
      "  [ and ]          switch runs",
      "  r                reload runs from disk",
      "  q or Esc         quit",
      "",
      "Discovered runs:",
      discovered,
    ].join("\n")
  );
}

function printSummary(runs: RunData[]): void {
  for (const run of runs) {
    const counts = run.counts;
    console.log(
      [
        `${run.name}  rows=${counts.total}`,
        `accept=${counts.decision.accept}`,
        `rewrite=${counts.decision.rewrite}`,
        `drop=${counts.decision.drop}`,
        `unknown=${counts.decision.unknown}`,
        `finalized=${counts.stage.finalized}`,
        `reviewed=${counts.stage.reviewed}`,
        `generated=${counts.stage.generated}`,
        `requested=${counts.stage.requested}`,
        `flagged=${counts.flagged}`,
      ].join("  ")
    );

    for (const filter of ["accept", "rewrite", "drop", "generated", "requested"] as const) {
      const entry = run.entries.find((item) => matchesFilter(item, filter));
      if (!entry) {
        continue;
      }
      const summary = entry.finalSummary || entry.candidateSummary || entry.dropReason || "(no summary yet)";
      console.log(
        `  [${filter}] ${entry.paperId} ${truncatePlain(entry.title, 64)} :: ${truncatePlain(summary, 76)}`
      );
    }
    console.log("");
  }
}

function loadRuns(explicitRunDir: string | undefined, rootDir: string): RunData[] {
  const dirs = explicitRunDir ? [explicitRunDir] : discoverRunDirectories(rootDir);
  const runs = dirs
    .filter((dir) => fs.existsSync(dir) && fs.statSync(dir).isDirectory())
    .map((dir) => loadRun(dir))
    .filter((run) => run.entries.length > 0 || run.files.length > 0);

  runs.sort((left, right) => safeMtime(right.dir) - safeMtime(left.dir));
  return runs;
}

function discoverRunDirectories(rootDir: string): string[] {
  if (!fs.existsSync(rootDir) || !fs.statSync(rootDir).isDirectory()) {
    return [];
  }
  return fs
    .readdirSync(rootDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(rootDir, entry.name));
}

function loadRun(dir: string): RunData {
  const artifacts: RunArtifacts = {
    manifest: readJsonMaybe(path.join(dir, "run_manifest.json")),
    sampleRows: readJsonlMaybe(path.join(dir, "sample_source_rows.jsonl")),
    generationRequests: readJsonlMaybe(path.join(dir, "generation_requests.jsonl")),
    generationOutputs: readJsonlMaybe(path.join(dir, "generation_outputs.jsonl")),
    reviewRequests: readJsonlMaybe(path.join(dir, "review_requests.jsonl")),
    reviewOutputs: readJsonlMaybe(path.join(dir, "review_outputs.jsonl")),
    trainingRows: readJsonlMaybe(path.join(dir, "training_rows.jsonl")),
    droppedRows: readJsonlMaybe(path.join(dir, "dropped_rows.jsonl")),
  };

  const files = ARTIFACT_FILE_ORDER.filter((file) => fs.existsSync(path.join(dir, file)));
  const idOrder = collectPaperIdsInOrder(artifacts);
  const sampleById = mapByPaperId(artifacts.sampleRows);
  const generationRequestById = mapByPaperId(artifacts.generationRequests);
  const generationOutputById = mapByPaperId(artifacts.generationOutputs);
  const reviewRequestById = mapByPaperId(artifacts.reviewRequests);
  const reviewOutputById = mapByPaperId(artifacts.reviewOutputs);
  const trainingById = mapByPaperId(artifacts.trainingRows);
  const droppedById = mapByPaperId(artifacts.droppedRows);

  const entries = idOrder.map((paperId, index) =>
    normalizeEntry({
      index,
      paperId,
      sampleRow: sampleById.get(paperId),
      generationRequest: generationRequestById.get(paperId),
      generationOutput: generationOutputById.get(paperId),
      reviewRequest: reviewRequestById.get(paperId),
      reviewOutput: reviewOutputById.get(paperId),
      trainingRow: trainingById.get(paperId),
      droppedRow: droppedById.get(paperId),
    })
  );

  return {
    name: path.basename(dir),
    dir,
    manifest: artifacts.manifest,
    entries,
    counts: buildRunCounts(entries),
    files,
  };
}

function normalizeEntry(parts: {
  index: number;
  paperId: string;
  sampleRow?: JsonObject;
  generationRequest?: JsonObject;
  generationOutput?: JsonObject;
  reviewRequest?: JsonObject;
  reviewOutput?: JsonObject;
  trainingRow?: JsonObject;
  droppedRow?: JsonObject;
}): RunEntry {
  const title = pickFirstString(
    readString(parts.trainingRow, "title"),
    readString(parts.droppedRow, "title"),
    readString(parts.sampleRow, "title"),
    extractTitleFromRequest(parts.generationRequest),
    extractTitleFromRequest(parts.reviewRequest)
  );
  const abstract = pickFirstString(
    readString(parts.trainingRow, "abstract"),
    readString(parts.droppedRow, "abstract"),
    readString(parts.sampleRow, "abstract"),
    extractAbstractFromRequest(parts.generationRequest),
    extractAbstractFromRequest(parts.reviewRequest)
  );
  const primaryCategory = pickFirstString(
    readString(parts.trainingRow, "primary_category"),
    readString(parts.droppedRow, "primary_category"),
    readString(parts.sampleRow, "primary_category")
  );
  const categories = pickFirstStringArray(
    readStringArray(parts.trainingRow, "categories"),
    readStringArray(parts.droppedRow, "categories"),
    readStringArray(parts.sampleRow, "categories")
  );
  const createdDate = pickFirstString(
    readString(parts.trainingRow, "created_date"),
    readString(parts.droppedRow, "created_date"),
    readString(parts.sampleRow, "created_date")
  );

  const candidateSummary = pickFirstString(
    readString(parts.generationOutput, "candidate_summary"),
    readNestedString(parts.generationOutput, "parsed", "summary"),
    readString(parts.reviewOutput, "candidate_summary"),
    readString(parts.reviewRequest, "candidate_summary"),
    readString(parts.droppedRow, "candidate_summary")
  );
  const finalSummary = pickFirstString(
    readString(parts.trainingRow, "summary"),
    readString(parts.reviewOutput, "summary"),
    readNestedString(parts.reviewOutput, "parsed", "summary")
  );
  const reviewDecision = normalizeDecision(
    pickFirstString(
      readString(parts.trainingRow, "review_decision"),
      readString(parts.droppedRow, "review_decision"),
      readString(parts.reviewOutput, "decision"),
      readNestedString(parts.reviewOutput, "parsed", "decision")
    )
  );
  const dropReason = pickFirstString(
    readString(parts.droppedRow, "drop_reason"),
    readString(parts.reviewOutput, "drop_reason"),
    readNestedString(parts.reviewOutput, "parsed", "drop_reason")
  );
  const reviewStage = inferReviewStage(parts);
  const changeKind = inferChangeKind(reviewDecision, candidateSummary, finalSummary);
  const candidateFlags = readStringArray(parts.generationOutput, "candidate_flags");
  const finalFlags = readStringArray(parts.reviewOutput, "summary_flags");
  const normalizedCandidateFlags =
    candidateFlags.length > 0 ? candidateFlags : summaryFlags(candidateSummary, title);
  const normalizedFinalFlags =
    finalFlags.length > 0 ? finalFlags : reviewDecision === "drop" ? [] : summaryFlags(finalSummary, title);
  const hasCandidate = candidateSummary.trim().length > 0;
  const hasFinal = finalSummary.trim().length > 0;

  const searchBlob = [
    parts.paperId,
    title,
    abstract,
    primaryCategory,
    categories.join(" "),
    candidateSummary,
    finalSummary,
    reviewDecision,
    reviewStage,
    changeKind,
    dropReason,
    normalizedCandidateFlags.join(" "),
    normalizedFinalFlags.join(" "),
  ]
    .join(" ")
    .toLowerCase();

  return {
    index: parts.index,
    paperId: parts.paperId,
    title,
    abstract,
    primaryCategory,
    categories,
    createdDate,
    reviewDecision,
    reviewStage,
    changeKind,
    candidateSummary,
    finalSummary,
    dropReason,
    candidateFlags: normalizedCandidateFlags,
    finalFlags: normalizedFinalFlags,
    hasCandidate,
    hasFinal,
    sampleRow: parts.sampleRow,
    generationRequest: parts.generationRequest,
    generationOutput: parts.generationOutput,
    reviewRequest: parts.reviewRequest,
    reviewOutput: parts.reviewOutput,
    trainingRow: parts.trainingRow,
    droppedRow: parts.droppedRow,
    searchBlob,
  };
}

function buildRunCounts(entries: RunEntry[]): RunCounts {
  const counts: RunCounts = {
    total: entries.length,
    decision: {
      accept: 0,
      rewrite: 0,
      drop: 0,
      unknown: 0,
    },
    stage: {
      finalized: 0,
      reviewed: 0,
      generated: 0,
      requested: 0,
      source_only: 0,
    },
    change: {
      accepted_unchanged: 0,
      rewritten_changed: 0,
      rewritten_unchanged: 0,
      dropped: 0,
      pending: 0,
    },
    flagged: 0,
  };

  for (const entry of entries) {
    counts.decision[entry.reviewDecision] += 1;
    counts.change[entry.changeKind] += 1;
    if (entry.reviewStage === "finalized") {
      counts.stage.finalized += 1;
    } else if (entry.reviewStage === "reviewed") {
      counts.stage.reviewed += 1;
    } else if (entry.reviewStage === "generated") {
      counts.stage.generated += 1;
    } else if (entry.generationRequest) {
      counts.stage.requested += 1;
    } else {
      counts.stage.source_only += 1;
    }
    if (entry.candidateFlags.length > 0 || entry.finalFlags.length > 0) {
      counts.flagged += 1;
    }
  }

  return counts;
}

function collectPaperIdsInOrder(artifacts: RunArtifacts): string[] {
  const seen = new Set<string>();
  const orderedIds: string[] = [];
  const sources = [
    artifacts.sampleRows,
    artifacts.generationRequests,
    artifacts.generationOutputs,
    artifacts.reviewRequests,
    artifacts.reviewOutputs,
    artifacts.trainingRows,
    artifacts.droppedRows,
  ];

  for (const rows of sources) {
    for (const row of rows) {
      const paperId = readString(row, "paper_id");
      if (!paperId || seen.has(paperId)) {
        continue;
      }
      seen.add(paperId);
      orderedIds.push(paperId);
    }
  }

  return orderedIds;
}

function mapByPaperId(rows: JsonObject[]): Map<string, JsonObject> {
  const map = new Map<string, JsonObject>();
  for (const row of rows) {
    const paperId = readString(row, "paper_id");
    if (paperId) {
      map.set(paperId, row);
    }
  }
  return map;
}

function normalizeDecision(value: string): ReviewDecision {
  const normalized = value.trim().toLowerCase();
  if (normalized === "accept" || normalized === "rewrite" || normalized === "drop") {
    return normalized;
  }
  return "unknown";
}

function inferReviewStage(parts: {
  sampleRow?: JsonObject;
  generationRequest?: JsonObject;
  generationOutput?: JsonObject;
  reviewOutput?: JsonObject;
  trainingRow?: JsonObject;
  droppedRow?: JsonObject;
}): ReviewStage {
  if (parts.trainingRow || parts.droppedRow) {
    return "finalized";
  }
  if (parts.reviewOutput) {
    return "reviewed";
  }
  if (parts.generationOutput) {
    return "generated";
  }
  if (parts.generationRequest || parts.sampleRow) {
    return "prepare_only";
  }
  return "prepare_only";
}

function inferChangeKind(
  reviewDecision: ReviewDecision,
  candidateSummary: string,
  finalSummary: string
): ChangeKind {
  const candidate = normalizeText(candidateSummary);
  const final = normalizeText(finalSummary);
  if (reviewDecision === "accept") {
    return "accepted_unchanged";
  }
  if (reviewDecision === "rewrite") {
    return candidate && final && candidate !== final ? "rewritten_changed" : "rewritten_unchanged";
  }
  if (reviewDecision === "drop") {
    return "dropped";
  }
  return "pending";
}

function readJsonMaybe(filePath: string): JsonObject | undefined {
  if (!fs.existsSync(filePath)) {
    return undefined;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as JsonObject;
  } catch {
    return { _error: `Could not parse ${path.basename(filePath)}` };
  }
}

function readJsonlMaybe(filePath: string): JsonObject[] {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  const content = fs.readFileSync(filePath, "utf8");
  const rows: JsonObject[] = [];
  for (const [lineIndex, rawLine] of content.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    try {
      rows.push(JSON.parse(line) as JsonObject);
    } catch {
      rows.push({
        paper_id: `parse-error-${lineIndex + 1}`,
        _error: `Could not parse line ${lineIndex + 1} in ${path.basename(filePath)}`,
        _raw: truncatePlain(line, 300),
      });
    }
  }
  return rows;
}

function readString(row: JsonObject | undefined, key: string): string {
  const value = row?.[key];
  return typeof value === "string" ? value : "";
}

function readNestedString(row: JsonObject | undefined, firstKey: string, secondKey: string): string {
  const nested = row?.[firstKey];
  if (!nested || typeof nested !== "object") {
    return "";
  }
  const value = (nested as JsonObject)[secondKey];
  return typeof value === "string" ? value : "";
}

function readStringArray(row: JsonObject | undefined, key: string): string[] {
  const value = row?.[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function pickFirstString(...values: string[]): string {
  return values.find((value) => value.trim().length > 0)?.trim() ?? "";
}

function pickFirstStringArray(...values: string[][]): string[] {
  return values.find((value) => value.length > 0) ?? [];
}

function messageContent(row: JsonObject | undefined, messageIndex: number): string {
  const messages = row?.messages;
  if (!Array.isArray(messages)) {
    return "";
  }
  const message = messages[messageIndex];
  if (!message || typeof message !== "object") {
    return "";
  }
  const content = (message as JsonObject).content;
  return typeof content === "string" ? content : "";
}

function extractTitleFromRequest(row: JsonObject | undefined): string {
  return extractPromptBlock(messageContent(row, 1), "TITLE:");
}

function extractAbstractFromRequest(row: JsonObject | undefined): string {
  return extractPromptBlock(messageContent(row, 1), "ABSTRACT:");
}

function extractPromptBlock(prompt: string, marker: string): string {
  const markerIndex = prompt.indexOf(marker);
  if (markerIndex === -1) {
    return "";
  }
  const afterMarker = prompt.slice(markerIndex + marker.length).trimStart();
  const nextMarkerIndex = afterMarker.search(/\n[A-Z][A-Z ]+:\n/);
  const block = nextMarkerIndex === -1 ? afterMarker : afterMarker.slice(0, nextMarkerIndex);
  return block.trim();
}

function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function wordCount(value: string): number {
  return normalizeText(value).match(WORD_RE)?.length ?? 0;
}

function titleOverlapRatio(summary: string, title: string): number {
  const summaryTokens = new Set((summary.match(WORD_RE) ?? []).map((token) => token.toLowerCase()));
  const titleTokens = new Set((title.match(WORD_RE) ?? []).map((token) => token.toLowerCase()));
  if (summaryTokens.size === 0 || titleTokens.size === 0) {
    return 0;
  }
  let overlap = 0;
  for (const token of summaryTokens) {
    if (titleTokens.has(token)) {
      overlap += 1;
    }
  }
  return overlap / summaryTokens.size;
}

function summaryFlags(summary: string, title: string): string[] {
  const normalized = normalizeText(summary).toLowerCase();
  if (!normalized) {
    return [];
  }
  const flags: string[] = [];
  const words = wordCount(summary);
  const sentenceMarkers = summary.split("").filter((char) => char === "." || char === "!" || char === "?").length;
  const overlap = titleOverlapRatio(summary, title);

  if (words < 8) {
    flags.push("too_short");
  }
  if (words > 18) {
    flags.push("too_long");
  }
  if (sentenceMarkers > 1) {
    flags.push("multiple_sentences");
  }
  if (BANNED_OPENINGS.some((prefix) => normalized.startsWith(prefix))) {
    flags.push("banned_opening");
  }
  if (overlap >= 0.75) {
    flags.push("high_title_overlap");
  }
  return flags;
}

function safeMtime(dir: string): number {
  try {
    return fs.statSync(dir).mtimeMs;
  } catch {
    return 0;
  }
}

function recomputeEntries(state: AppState): void {
  const run = currentRun(state);
  const query = state.query.trim().toLowerCase();
  state.filteredEntries = run.entries.filter((entry) => {
    const filterMatches = matchesFilter(entry, state.filter);
    const queryMatches = !query || entry.searchBlob.includes(query);
    return filterMatches && queryMatches;
  });
  state.selectedIndex = clamp(state.selectedIndex, 0, Math.max(0, state.filteredEntries.length - 1));
  state.listScroll = 0;
  state.detailScroll = 0;
}

function matchesFilter(entry: RunEntry, filter: FilterKey): boolean {
  switch (filter) {
    case "all":
      return true;
    case "accept":
      return entry.reviewDecision === "accept";
    case "rewrite":
      return entry.reviewDecision === "rewrite";
    case "drop":
      return entry.reviewDecision === "drop";
    case "rewritten_unchanged":
      return entry.changeKind === "rewritten_unchanged";
    case "flagged":
      return entry.candidateFlags.length > 0 || entry.finalFlags.length > 0;
    case "reviewed":
      return entry.reviewStage === "reviewed";
    case "generated":
      return entry.reviewStage === "generated";
    case "requested":
      return entry.reviewStage === "prepare_only" && Boolean(entry.generationRequest);
    case "source_only":
      return entry.reviewStage === "prepare_only" && !entry.generationRequest;
  }
}

function currentRun(state: AppState): RunData {
  return state.runs[state.runIndex];
}

function currentEntry(state: AppState): RunEntry | undefined {
  return state.filteredEntries[state.selectedIndex];
}

function startInteractiveSession(state: AppState): void {
  readline.emitKeypressEvents(process.stdin);
  process.stdin.setRawMode(true);
  process.stdin.resume();

  let cleanedUp = false;
  const cleanup = () => {
    if (cleanedUp) {
      return;
    }
    cleanedUp = true;
    process.stdout.write(ansi.reset + ansi.showCursor + ansi.altScreenOff);
    process.stdout.write("\n");
    process.stdin.setRawMode(false);
    process.stdin.pause();
  };

  process.stdout.write(ansi.altScreenOn + ansi.hideCursor);
  render(state);

  const exit = () => {
    if (state.shouldExit) {
      cleanup();
      process.exit(0);
    }
    state.shouldExit = true;
    cleanup();
    process.exit(0);
  };

  process.on("SIGINT", exit);
  process.on("uncaughtException", (error) => {
    cleanup();
    console.error(error);
    process.exitCode = 1;
  });

  process.stdout.on("resize", () => render(state));
  process.stdin.on("keypress", (str, key) => {
    handleKeypress(state, str, key);
    if (state.shouldExit) {
      exit();
      return;
    }
    render(state);
  });
}

function handleKeypress(state: AppState, str: string, key: readline.Key): void {
  if (state.mode === "search") {
    handleSearchKeypress(state, str, key);
    return;
  }

  switch (key.name) {
    case "q":
      state.shouldExit = true;
      return;
    case "escape":
      state.shouldExit = true;
      return;
    case "up":
    case "k":
      moveSelection(state, -1);
      return;
    case "down":
    case "j":
      moveSelection(state, 1);
      return;
    case "left":
      cycleTab(state, -1);
      return;
    case "right":
      cycleTab(state, 1);
      return;
    case "tab":
      cycleTab(state, key.shift ? -1 : 1);
      return;
    case "pageup":
      state.detailScroll = Math.max(0, state.detailScroll - detailPageSize());
      return;
    case "pagedown":
      state.detailScroll += detailPageSize();
      return;
    case "space":
      state.detailScroll += detailPageSize();
      return;
    case "home":
      state.selectedIndex = 0;
      state.listScroll = 0;
      state.detailScroll = 0;
      return;
    case "end":
      state.selectedIndex = Math.max(0, state.filteredEntries.length - 1);
      state.detailScroll = 0;
      return;
    case "r":
      reloadRuns(state);
      return;
    case "i":
      state.showRunInfo = !state.showRunInfo;
      return;
    case "c":
      state.query = "";
      state.draftQuery = "";
      recomputeEntries(state);
      return;
    default:
      break;
  }

  if (str === "/") {
    state.mode = "search";
    state.draftQuery = state.query;
    return;
  }
  if (str === "f") {
    const currentIndex = FILTER_ORDER.indexOf(state.filter);
    state.filter = FILTER_ORDER[(currentIndex + 1) % FILTER_ORDER.length];
    recomputeEntries(state);
    return;
  }
  if (str === "[") {
    switchRun(state, -1);
    return;
  }
  if (str === "]") {
    switchRun(state, 1);
    return;
  }
  if (str === "1") {
    state.tab = "overview";
    state.detailScroll = 0;
    return;
  }
  if (str === "2") {
    state.tab = "compare";
    state.detailScroll = 0;
    return;
  }
  if (str === "3") {
    state.tab = "generator";
    state.detailScroll = 0;
    return;
  }
  if (str === "4") {
    state.tab = "reviewer";
    state.detailScroll = 0;
    return;
  }
  if (str === "5") {
    state.tab = "raw";
    state.detailScroll = 0;
    return;
  }
  if (str === "g") {
    state.selectedIndex = 0;
    state.listScroll = 0;
    state.detailScroll = 0;
    return;
  }
  if (str === "G") {
    state.selectedIndex = Math.max(0, state.filteredEntries.length - 1);
    state.detailScroll = 0;
    return;
  }
  if (str === "h") {
    cycleTab(state, -1);
    return;
  }
  if (str === "l") {
    cycleTab(state, 1);
    return;
  }
  if (str === "b") {
    state.detailScroll = Math.max(0, state.detailScroll - detailPageSize());
    return;
  }
  if (key.ctrl && key.name === "d") {
    state.detailScroll += detailPageSize();
    return;
  }
  if (key.ctrl && key.name === "u") {
    state.detailScroll = Math.max(0, state.detailScroll - detailPageSize());
  }
}

function handleSearchKeypress(state: AppState, str: string, key: readline.Key): void {
  if (key.name === "escape") {
    state.mode = "browse";
    state.draftQuery = state.query;
    return;
  }
  if (key.name === "return" || key.name === "enter") {
    state.query = state.draftQuery;
    state.mode = "browse";
    recomputeEntries(state);
    return;
  }
  if (key.name === "backspace") {
    state.draftQuery = state.draftQuery.slice(0, -1);
    return;
  }
  if (key.ctrl && key.name === "u") {
    state.draftQuery = "";
    return;
  }
  if (str && !key.ctrl && !key.meta && str >= " ") {
    state.draftQuery += str;
  }
}

function reloadRuns(state: AppState): void {
  const existingDir = currentRun(state).dir;
  detailCache.clear();
  state.runs = loadRuns(state.scopedRunDir, state.rootDir);
  const newIndex = state.runs.findIndex((run) => run.dir === existingDir);
  state.runIndex = Math.max(0, newIndex);
  recomputeEntries(state);
}

function switchRun(state: AppState, direction: -1 | 1): void {
  if (state.runs.length < 2) {
    return;
  }
  state.runIndex = (state.runIndex + direction + state.runs.length) % state.runs.length;
  state.selectedIndex = 0;
  state.detailScroll = 0;
  recomputeEntries(state);
}

function moveSelection(state: AppState, delta: number): void {
  if (state.filteredEntries.length === 0) {
    return;
  }
  const previousIndex = state.selectedIndex;
  state.selectedIndex = clamp(state.selectedIndex + delta, 0, state.filteredEntries.length - 1);
  if (state.selectedIndex !== previousIndex) {
    state.detailScroll = 0;
  }
}

function cycleTab(state: AppState, direction: -1 | 1): void {
  const currentIndex = TAB_ORDER.indexOf(state.tab);
  const nextIndex = (currentIndex + direction + TAB_ORDER.length) % TAB_ORDER.length;
  state.tab = TAB_ORDER[nextIndex];
  state.detailScroll = 0;
}

function render(state: AppState): void {
  const width = process.stdout.columns || 120;
  const height = process.stdout.rows || 40;
  const run = currentRun(state);
  const entry = currentEntry(state);

  const headerLines = buildHeaderLines(state, width);
  const footerLines = buildFooterLines(state, width);
  const availableHeight = Math.max(6, height - headerLines.length - footerLines.length);
  const listWidth = clamp(Math.floor(width * 0.28), 26, 42);
  const detailWidth = Math.max(20, width - listWidth - 3);

  syncListScroll(state, availableHeight);
  const listLines = buildListLines(state, listWidth, availableHeight);
  const detailLines = buildDetailLines(run, entry, state.tab, detailWidth);
  const maxDetailScroll = Math.max(0, detailLines.length - availableHeight);
  state.detailScroll = clamp(state.detailScroll, 0, maxDetailScroll);
  state.detailLineCount = detailLines.length;
  state.detailViewportHeight = availableHeight;
  const detailSlice = detailLines.slice(state.detailScroll, state.detailScroll + availableHeight);

  const output: string[] = [];
  output.push(ansi.clear);
  output.push(...headerLines);

  for (let row = 0; row < availableHeight; row += 1) {
    const listLine = padRight(listLines[row] ?? "", listWidth);
    const detailLine = padRight(detailSlice[row] ?? "", detailWidth);
    output.push(`${listLine} ${formatDivider()} ${detailLine}`);
  }

  output.push(...footerLines);
  process.stdout.write(output.join("\n") + ansi.reset);
}

function buildHeaderLines(state: AppState, width: number): string[] {
  const run = currentRun(state);
  const manifestMode = readString(run.manifest, "mode") || "derived";
  const compactLine = [
    `${ansi.bold}${run.name}${ansi.reset}`,
    `${ansi.dim}${manifestMode}${ansi.reset}`,
    `tab=${state.tab}`,
    `filter=${state.filter}`,
    `query=${state.query || "(none)"}`,
  ].join("  ");
  const countsLine = [
    `${ansi.green}KEEP${ansi.reset}:${run.counts.decision.accept}`,
    `${ansi.yellow}RWRT${ansi.reset}:${run.counts.decision.rewrite}`,
    `${ansi.red}DROP${ansi.reset}:${run.counts.decision.drop}`,
    `${ansi.magenta}FLAG${ansi.reset}:${run.counts.flagged}`,
  ].join("  ");

  const lines = [truncateAnsi(compactLine, width), truncateAnsi(countsLine, width)];

  if (state.showRunInfo) {
    const infoLine1 = [
      `dir=${run.dir}`,
      `gen_prompt=${readString(run.manifest, "generator_prompt_version") || "-"}`,
      `rev_prompt=${readString(run.manifest, "review_prompt_version") || "-"}`,
    ].join("  ");
    const infoLine2 = [
      `gen_model=${readString(run.manifest, "generator_model") || "-"}`,
      `rev_model=${readString(run.manifest, "reviewer_model") || "-"}`,
      `files=${run.files.join(", ") || "(none)"}`,
    ].join("  ");
    const infoLine3 = [
      `manifest_review=${formatCountObject(readValue(run.manifest, "review_decisions"))}`,
      `gen_flags=${formatCountObject(readValue(run.manifest, "generator_flag_counts"))}`,
      `final_flags=${formatCountObject(readValue(run.manifest, "review_flag_counts"))}`,
    ].join("  ");
    lines.push(truncateAnsi(infoLine1, width));
    lines.push(truncateAnsi(infoLine2, width));
    lines.push(truncateAnsi(infoLine3, width));
  }

  lines.push(repeat("-", width));
  return lines;
}

function buildFooterLines(state: AppState, width: number): string[] {
  const searchPrompt =
    state.mode === "search"
      ? `${ansi.bold}/${state.draftQuery}${ansi.reset}`
      : "q/esc quit | / search | c clear | f filter | i info | 1-5 or ←→/h l tabs | pgdn/space b/pgup scroll | [ ] runs | r reload";
  const entry = currentEntry(state);
  const detailStatus =
    state.detailLineCount > state.detailViewportHeight
      ? `  detail ${state.detailScroll + 1}-${Math.min(state.detailScroll + state.detailViewportHeight, state.detailLineCount)}/${state.detailLineCount}`
      : "";
  const status = entry
    ? [
        `row ${state.selectedIndex + 1}/${state.filteredEntries.length}`,
        entry.paperId,
        `decision=${entry.reviewDecision}`,
        `stage=${entry.reviewStage}`,
      ].join("  ") + detailStatus
    : `row 0/${state.filteredEntries.length}`;
  return [repeat("-", width), truncateAnsi(status, width), truncateAnsi(searchPrompt, width)];
}

function buildListLines(state: AppState, width: number, height: number): string[] {
  if (state.filteredEntries.length === 0) {
    return [padRight("No rows match the current filter/query.", width)];
  }

  const lines: string[] = [];
  const visibleCapacity = Math.max(1, height);
  const visible = state.filteredEntries.slice(state.listScroll, state.listScroll + visibleCapacity);
  for (const [offset, entry] of visible.entries()) {
    const absoluteIndex = state.listScroll + offset;
    const isSelected = absoluteIndex === state.selectedIndex;
    lines.push(renderListLine(entry, width, isSelected));
  }
  return lines.slice(0, height);
}

function renderListLine(entry: RunEntry, width: number, isSelected: boolean): string {
  const marker = isSelected ? ">" : " ";
  const showFlag = width >= 40;
  const decisionWidth = 4;
  const idWidth = width >= 36 ? 12 : 10;
  const hasFlag = entry.finalFlags.length > 0 || entry.candidateFlags.length > 0;
  const reserved =
    1 + 1 + decisionWidth + 1 + idWidth + 1 + (showFlag && hasFlag ? 2 : 0);
  const titleWidth = Math.max(8, width - reserved);

  const decisionText = fitPlain(decisionLabel(entry), decisionWidth);
  const idText = fitPlain(entry.paperId, idWidth);
  const titleText = fitPlain(entry.title, titleWidth);
  const flagText = showFlag && hasFlag ? "!" : "";

  const plain =
    `${marker} ${decisionText} ${idText} ${titleText}` +
    (flagText ? ` ${flagText}` : "");

  if (isSelected) {
    return padRight(
      `${ansi.bold}${ansi.cyan}${marker}${ansi.reset} ${colorize(decisionText, decisionColor(entry))} ${ansi.cyan}${ansi.bold}${idText}${ansi.reset} ${ansi.cyan}${ansi.bold}${titleText}${ansi.reset}` +
        (flagText ? ` ${ansi.red}${flagText}${ansi.reset}` : ""),
      width
    );
  }

  let line =
    `${marker} ${colorize(decisionText, decisionColor(entry))} ${ansi.dim}${idText}${ansi.reset} ${titleText}` +
    (flagText ? ` ${ansi.red}${flagText}${ansi.reset}` : "");

  return padRight(line, width);
}

function buildDetailLines(run: RunData, entry: RunEntry | undefined, tab: DetailTab, width: number): string[] {
  if (!entry) {
    return wrapBlock("No entry selected.", width);
  }

  const cacheKey = `${run.name}:${entry.paperId}:${tab}:${width}`;
  const cached = detailCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const sections: string[] = [];
  if (tab === "overview") {
    sections.push(sectionHeading("Final summary"));
    sections.push(entry.finalSummary || "(none yet)");
    sections.push("");
    sections.push(sectionHeading("Status"));
    sections.push(`Decision: ${entry.reviewDecision}`);
    sections.push(`Stage: ${entry.reviewStage}`);
    sections.push(
      `Flags: candidate=${entry.candidateFlags.join(", ") || "(none)"} | final=${entry.finalFlags.join(", ") || "(none)"}`
    );
    if (entry.dropReason) {
      sections.push(`Drop reason: ${entry.dropReason}`);
    }
    sections.push("");
    sections.push(sectionHeading("Paper"));
    sections.push(`ID: ${entry.paperId}`);
    sections.push(`Category: ${entry.primaryCategory || "-"}    Created: ${entry.createdDate || "-"}`);
    if (entry.categories.length > 0) {
      sections.push(`All categories: ${entry.categories.join(", ")}`);
    }
    sections.push("");
    sections.push(sectionHeading("Title"));
    sections.push(entry.title || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Abstract"));
    sections.push(entry.abstract || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Candidate summary"));
    sections.push(entry.hasCandidate ? "Use the Compare tab (4) for candidate/final inspection." : "(missing)");
  } else if (tab === "generator") {
    sections.push(sectionHeading("Generator"));
    sections.push(`Prompt version: ${pickFirstString(readString(entry.generationRequest, "prompt_version"), readString(entry.generationOutput, "prompt_version")) || "-"}`);
    sections.push(`Model: ${pickFirstString(readString(entry.generationOutput, "model"), readString(entry.generationRequest, "model")) || "-"}`);
    sections.push(`Response id: ${readString(entry.generationOutput, "api_response_id") || "-"}`);
    sections.push(`Candidate flags: ${entry.candidateFlags.join(", ") || "(none)"}`);
    sections.push("");
    sections.push(sectionHeading("Candidate summary"));
    sections.push(entry.candidateSummary || "(none yet)");
    sections.push("");
    sections.push(sectionHeading("System prompt"));
    sections.push(messageContent(entry.generationRequest, 0) || "(missing)");
    sections.push("");
    sections.push(sectionHeading("User prompt"));
    sections.push(messageContent(entry.generationRequest, 1) || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Raw output"));
    sections.push(readString(entry.generationOutput, "raw_text") || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Parsed output"));
    sections.push(jsonPretty(readValue(entry.generationOutput, "parsed")));
  } else if (tab === "reviewer") {
    sections.push(sectionHeading("Reviewer"));
    sections.push(`Prompt version: ${pickFirstString(readString(entry.reviewRequest, "prompt_version"), readString(entry.reviewOutput, "prompt_version")) || "-"}`);
    sections.push(`Model: ${pickFirstString(readString(entry.reviewOutput, "model"), readString(entry.reviewRequest, "model")) || "-"}`);
    sections.push(`Response id: ${readString(entry.reviewOutput, "api_response_id") || "-"}`);
    sections.push(`Decision: ${entry.reviewDecision}`);
    sections.push(`Final flags: ${entry.finalFlags.join(", ") || "(none)"}`);
    if (entry.dropReason) {
      sections.push(`Drop reason: ${entry.dropReason}`);
    }
    sections.push("");
    sections.push(sectionHeading("System prompt"));
    sections.push(messageContent(entry.reviewRequest, 0) || "(missing)");
    sections.push("");
    sections.push(sectionHeading("User prompt"));
    sections.push(messageContent(entry.reviewRequest, 1) || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Raw output"));
    sections.push(readString(entry.reviewOutput, "raw_text") || "(missing)");
    sections.push("");
    sections.push(sectionHeading("Parsed output"));
    sections.push(jsonPretty(readValue(entry.reviewOutput, "parsed")));
  } else if (tab === "compare") {
    sections.push(sectionHeading("Compare"));
    sections.push(`Decision: ${entry.reviewDecision}`);
    sections.push("");
    sections.push(sectionHeading("Final summary"));
    sections.push(entry.finalSummary || "(none yet)");
    sections.push("");
    sections.push(sectionHeading("Candidate summary"));
    sections.push(entry.candidateSummary || "(none yet)");
    sections.push("");
    sections.push(`Candidate flags: ${entry.candidateFlags.join(", ") || "(none)"}`);
    sections.push(`Final flags: ${entry.finalFlags.join(", ") || "(none)"}`);
    if (entry.dropReason) {
      sections.push(`Drop reason: ${entry.dropReason}`);
    }
  } else {
    sections.push(sectionHeading("Manifest"));
    sections.push(jsonPretty(run.manifest));
    sections.push("");
    sections.push(sectionHeading("Sample row"));
    sections.push(jsonPretty(entry.sampleRow));
    sections.push("");
    sections.push(sectionHeading("Generation request"));
    sections.push(jsonPretty(entry.generationRequest));
    sections.push("");
    sections.push(sectionHeading("Generation output"));
    sections.push(jsonPretty(entry.generationOutput));
    sections.push("");
    sections.push(sectionHeading("Review request"));
    sections.push(jsonPretty(entry.reviewRequest));
    sections.push("");
    sections.push(sectionHeading("Review output"));
    sections.push(jsonPretty(entry.reviewOutput));
    sections.push("");
    sections.push(sectionHeading("Training row"));
    sections.push(jsonPretty(entry.trainingRow));
    sections.push("");
    sections.push(sectionHeading("Dropped row"));
    sections.push(jsonPretty(entry.droppedRow));
  }

  const rendered = sections.flatMap((section) => wrapBlock(section, width));
  detailCache.set(cacheKey, rendered);
  return rendered;
}

function readValue(row: JsonObject | undefined, key: string): unknown {
  return row?.[key];
}

function jsonPretty(value: unknown): string {
  if (value === undefined) {
    return "(missing)";
  }
  return JSON.stringify(value, null, 2);
}

function wrapBlock(text: string, width: number): string[] {
  if (!text) {
    return [""];
  }
  const maxWidth = Math.max(8, width);
  const result: string[] = [];
  for (const paragraph of text.split("\n")) {
    const trimmed = paragraph.replace(/\t/g, "  ");
    if (!trimmed) {
      result.push("");
      continue;
    }
    let remaining = trimmed;
    while (remaining.length > maxWidth) {
      const slice = remaining.slice(0, maxWidth + 1);
      const breakIndex = Math.max(slice.lastIndexOf(" "), slice.lastIndexOf("/"), slice.lastIndexOf("-"));
      const cutAt = breakIndex > Math.floor(maxWidth * 0.4) ? breakIndex : maxWidth;
      result.push(remaining.slice(0, cutAt).trimEnd());
      remaining = remaining.slice(cutAt).trimStart();
    }
    result.push(remaining);
  }
  return result;
}

function syncListScroll(state: AppState, availableHeight: number): void {
  const visibleCapacity = Math.max(1, availableHeight);
  const visibleTop = state.listScroll;
  const visibleBottom = visibleTop + visibleCapacity - 1;
  if (state.selectedIndex < visibleTop) {
    state.listScroll = state.selectedIndex;
  } else if (state.selectedIndex > visibleBottom) {
    state.listScroll = Math.max(0, state.selectedIndex - visibleCapacity + 1);
  }
}

function detailPageSize(): number {
  const height = process.stdout.rows || 40;
  return Math.max(4, height - 12);
}

function decisionLabel(entry: RunEntry): string {
  if (entry.reviewDecision === "accept") {
    return "KEEP";
  }
  if (entry.reviewDecision === "rewrite") {
    return "RWRT";
  }
  if (entry.reviewDecision === "drop") {
    return "DROP";
  }
  if (entry.reviewStage === "reviewed") {
    return "REVW";
  }
  if (entry.reviewStage === "generated") {
    return "GEN ";
  }
  if (entry.generationRequest) {
    return "REQ ";
  }
  return "SRC ";
}

function decisionColor(entry: RunEntry): string {
  if (entry.reviewDecision === "accept") {
    return ansi.green;
  }
  if (entry.reviewDecision === "rewrite") {
    return ansi.yellow;
  }
  if (entry.reviewDecision === "drop") {
    return ansi.red;
  }
  if (entry.reviewStage === "reviewed") {
    return ansi.magenta;
  }
  if (entry.reviewStage === "generated") {
    return ansi.blue;
  }
  if (entry.generationRequest) {
    return ansi.cyan;
  }
  return "";
}

function formatDivider(): string {
  return `${ansi.dim}|${ansi.reset}`;
}

function sectionHeading(label: string): string {
  return `${ansi.bold}${ansi.blue}${label}${ansi.reset}`;
}

function formatCountObject(value: unknown): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return "(none)";
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, count]) => typeof count === "number")
    .sort((left, right) => String(left[0]).localeCompare(String(right[0])))
    .map(([key, count]) => `${key}:${count}`);
  return entries.length > 0 ? entries.join(",") : "(none)";
}

function colorize(text: string, color: string): string {
  return color ? `${color}${text}${ansi.reset}` : text;
}

function fitPlain(text: string, width: number): string {
  const truncated = truncatePlain(text, width);
  if (truncated.length >= width) {
    return truncated;
  }
  return truncated + " ".repeat(width - truncated.length);
}

function truncatePlain(text: string, width: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= width) {
    return clean;
  }
  return `${clean.slice(0, Math.max(0, width - 1)).trimEnd()}…`;
}

function truncateAnsi(text: string, width: number): string {
  return truncatePlain(stripAnsi(text), width);
}

function stripAnsi(text: string): string {
  return text.replace(/\u001b\[[0-9;?]*[A-Za-z]/g, "");
}

function repeat(char: string, count: number): string {
  return char.repeat(Math.max(0, count));
}

function padRight(text: string, width: number): string {
  const plainLength = stripAnsi(text).length;
  if (plainLength >= width) {
    return truncateAnsi(text, width);
  }
  return text + " ".repeat(width - plainLength);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

main();
