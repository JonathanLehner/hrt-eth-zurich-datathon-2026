import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { asyncBufferFromFile, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";

const DEFAULT_FILE = "data/headlines_seen_train.parquet";
const DEFAULT_MODEL = "gemini-2.5-flash";
const DEFAULT_CONCURRENCY = 3;
const DEFAULT_RETRIES = 3;
const DEFAULT_OUTPUT_DIR = "analysis";

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const env = await loadDotEnv(path.resolve(process.cwd(), ".env"));
  const apiKey = process.env.GEMINI_API_KEY ?? env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY in the environment.");
  }

  const filePath = path.resolve(process.cwd(), options.file);
  const sessions = await loadSessions(filePath);
  const selectedSessions = filterSessions(sessions, options);

  if (selectedSessions.length === 0) {
    throw new Error("No sessions matched the provided filters.");
  }

  console.log(
    `Analyzing ${selectedSessions.length} session(s) from ${path.relative(process.cwd(), filePath)} with ${options.model}`,
  );

  const startedAt = new Date().toISOString();
  const results = await mapWithConcurrency(
    selectedSessions,
    options.concurrency,
    async ([sessionId, headlines], index) => {
      console.log(`[${index + 1}/${selectedSessions.length}] session ${sessionId} (${headlines.length} headlines)`);
      return analyzeSession({
        apiKey,
        sessionId,
        headlines,
        model: options.model,
        retries: options.retries,
      });
    },
  );

  const outputPath = path.resolve(
    process.cwd(),
    options.output ?? defaultOutputPath(filePath),
  );

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(
    outputPath,
    JSON.stringify(
      {
        generatedAt: startedAt,
        sourceFile: path.relative(process.cwd(), filePath),
        model: options.model,
        sessionCount: results.length,
        results,
      },
      null,
      2,
    ),
  );

  console.log(`Saved analysis to ${path.relative(process.cwd(), outputPath)}`);
}

async function loadDotEnv(filePath) {
  try {
    const contents = await readFile(filePath, "utf8");
    return parseDotEnv(contents);
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return {};
    }
    throw error;
  }
}

function parseDotEnv(contents) {
  const values = {};

  for (const rawLine of contents.split(/\r?\n/u)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    const separatorIndex = line.indexOf("=");
    if (separatorIndex === -1) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    let value = line.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    if (key) {
      values[key] = value;
    }
  }

  return values;
}

function parseArgs(args) {
  const options = {
    file: DEFAULT_FILE,
    model: DEFAULT_MODEL,
    concurrency: DEFAULT_CONCURRENCY,
    retries: DEFAULT_RETRIES,
    output: null,
    sessionIds: [],
    maxSessions: null,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const value = args[index + 1];

    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    if (arg === "--file" || arg === "-f") {
      options.file = requireValue(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--model" || arg === "-m") {
      options.model = requireValue(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--output" || arg === "-o") {
      options.output = requireValue(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--session" || arg === "-s") {
      options.sessionIds.push(requireValue(arg, value));
      index += 1;
      continue;
    }

    if (arg === "--max-sessions") {
      options.maxSessions = parsePositiveInteger(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--concurrency" || arg === "-c") {
      options.concurrency = parsePositiveInteger(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--retries") {
      options.retries = parsePositiveInteger(arg, value);
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  npm run analyze-headlines -- --file data/headlines_seen_train.parquet
  npm run analyze-headlines -- --session 0 --session 1
  npm run analyze-headlines -- --max-sessions 25 --concurrency 2

Required:
  Set GEMINI_API_KEY in your environment.

Options:
  --file, -f           Headlines parquet file to analyze
  --model, -m          Gemini model to use (default: ${DEFAULT_MODEL})
  --session, -s        Analyze only a specific session; repeatable
  --max-sessions       Limit how many sessions to analyze
  --concurrency, -c    Parallel Gemini requests (default: ${DEFAULT_CONCURRENCY})
  --retries            Retries per session on API failure (default: ${DEFAULT_RETRIES})
  --output, -o         Output JSON path (default: analysis/<input>.gemini.json)`);
}

function requireValue(flag, value) {
  if (!value || value.startsWith("--")) {
    throw new Error(`Expected a value after ${flag}.`);
  }
  return value;
}

function parsePositiveInteger(flag, value) {
  const parsed = Number.parseInt(requireValue(flag, value), 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return parsed;
}

async function loadSessions(filePath) {
  const file = await asyncBufferFromFile(filePath);
  const rows = await parquetReadObjects({ file, compressors });
  const sessions = new Map();

  for (const row of rows) {
    const sessionId = String(row.session);
    const headline = {
      barIx: toComparableNumber(row.bar_ix),
      headline: row.headline,
    };

    if (!sessions.has(sessionId)) {
      sessions.set(sessionId, []);
    }

    sessions.get(sessionId).push(headline);
  }

  for (const headlines of sessions.values()) {
    headlines.sort((left, right) => left.barIx - right.barIx);
  }

  return sessions;
}

function filterSessions(sessions, options) {
  let entries = [...sessions.entries()].sort((left, right) => Number(left[0]) - Number(right[0]));

  if (options.sessionIds.length > 0) {
    const selected = new Set(options.sessionIds);
    entries = entries.filter(([sessionId]) => selected.has(sessionId));
  }

  if (options.maxSessions !== null) {
    entries = entries.slice(0, options.maxSessions);
  }

  return entries;
}

function toComparableNumber(value) {
  if (typeof value === "bigint") {
    return Number(value);
  }
  return Number(value);
}

async function analyzeSession({ apiKey, sessionId, headlines, model, retries }) {
  const prompt = buildPrompt(sessionId, headlines);
  const raw = await callGeminiWithRetry({
    apiKey,
    model,
    prompt,
    retries,
  });

  const parsed = JSON.parse(raw);
  return {
    session: sessionId,
    headlineCount: headlines.length,
    analysis: parsed,
  };
}

function buildPrompt(sessionId, headlines) {
  const formattedHeadlines = headlines
    .map(({ barIx, headline }) => `- bar ${barIx}: ${headline}`)
    .join("\n");

  return `You are analyzing synthetic financial headlines for a single hidden company session.

Goal:
- Infer which mentioned company name is most likely the real company being tracked in this session.
- The chosen company must be one of the company names that appear in the headlines.
- Prefer the company whose headlines form the most coherent repeated narrative across time.
- If the evidence is weak, say so clearly and lower confidence.

Return strict JSON only.

Session: ${sessionId}
Headlines:
${formattedHeadlines}`;
}

async function callGeminiWithRetry({ apiKey, model, prompt, retries }) {
  let attempt = 0;
  let lastError;

  while (attempt <= retries) {
    try {
      return await callGemini({ apiKey, model, prompt });
    } catch (error) {
      lastError = error;
      if (attempt === retries) {
        break;
      }
      const delayMs = 1000 * 2 ** attempt;
      await sleep(delayMs);
    }
    attempt += 1;
  }

  throw lastError;
}

async function callGemini({ apiKey, model, prompt }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      systemInstruction: {
        role: "system",
        parts: [
          {
            text: "You classify headlines and return strict JSON that matches the provided schema.",
          },
        ],
      },
      contents: [
        {
          role: "user",
          parts: [{ text: prompt }],
        },
      ],
      generationConfig: {
        temperature: 0.2,
        responseMimeType: "application/json",
        responseJsonSchema: responseSchema(),
      },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini API error ${response.status}: ${body}`);
  }

  const payload = await response.json();
  const text = extractResponseText(payload);

  if (!text) {
    throw new Error(`Gemini API returned no text: ${JSON.stringify(payload)}`);
  }

  return text;
}

function extractResponseText(payload) {
  const parts = payload?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) {
    return "";
  }

  return parts
    .map((part) => part.text ?? "")
    .join("")
    .trim();
}

function responseSchema() {
  return {
    type: "object",
    properties: {
      likelyCompany: {
        type: ["string", "null"],
        description: "The company from the headlines that is most likely the real tracked company.",
      },
      confidence: {
        type: "number",
        description: "A value from 0 to 1 reflecting how confident the choice is.",
      },
      industry: {
        type: ["string", "null"],
        description: "The inferred main industry or sector for the selected company.",
      },
      summary: {
        type: "string",
        description: "Brief explanation of why this company is the best fit.",
      },
      evidenceHeadlines: {
        type: "array",
        description: "Short list of the strongest supporting headlines, copied verbatim.",
        items: {
          type: "string",
        },
        minItems: 1,
        maxItems: 5,
      },
      alternatives: {
        type: "array",
        description: "Other plausible companies from the session and why they were rejected.",
        items: {
          type: "object",
          properties: {
            company: { type: "string" },
            whyNot: { type: "string" },
          },
          required: ["company", "whyNot"],
          additionalProperties: false,
        },
        maxItems: 5,
      },
    },
    required: [
      "likelyCompany",
      "confidence",
      "industry",
      "summary",
      "evidenceHeadlines",
      "alternatives",
    ],
    additionalProperties: false,
  };
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const results = new Array(items.length);
  let currentIndex = 0;

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (currentIndex < items.length) {
      const workIndex = currentIndex;
      currentIndex += 1;
      results[workIndex] = await mapper(items[workIndex], workIndex);
    }
  });

  await Promise.all(workers);
  return results;
}

function defaultOutputPath(filePath) {
  const parsed = path.parse(filePath);
  return path.join(DEFAULT_OUTPUT_DIR, `${parsed.name}.gemini.json`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
