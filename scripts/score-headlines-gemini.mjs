import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const DEFAULT_MODEL = "gemini-2.5-flash";
const DEFAULT_INPUT_DIR = "analysis/company-ids";
const DEFAULT_CONCURRENCY = 100;
const DEFAULT_RETRIES = 3;
const DEFAULT_REQUEST_DELAY_MS = 250;

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const env = await loadDotEnv(path.resolve(process.cwd(), ".env"));
  const apiKey = process.env.GEMINI_API_KEY ?? env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY in the environment or .env.");
  }

  const inputDir = path.resolve(process.cwd(), options.inputDir);
  const sessionFiles = await resolveSessionFiles(inputDir, options);

  if (sessionFiles.length === 0) {
    throw new Error("No session-*.json files matched the provided filters.");
  }

  const filteredSessionFiles = await filterAlreadyScoredFiles(sessionFiles, options.force);

  if (filteredSessionFiles.length === 0) {
    console.log("No unscored session files found. Use --force to rescore existing files.");
    return;
  }

  console.log(
    `Scoring headlines in ${filteredSessionFiles.length} session file(s) from ${path.relative(process.cwd(), inputDir)} with ${options.model} (concurrency ${options.concurrency}, delay ${options.requestDelayMs}ms)`,
  );

  const scheduler = createRequestScheduler(options.requestDelayMs);
  const progress = createProgressBar(filteredSessionFiles.length);
  await mapWithConcurrency(filteredSessionFiles, options.concurrency, async (filePath, index) => {
    const relativePath = path.relative(process.cwd(), filePath);
    const sessionData = JSON.parse(await readFile(filePath, "utf8"));
    const updated = await scoreSessionFile({
      apiKey,
      model: options.model,
      retries: options.retries,
      scheduler,
      sessionData,
    });
    await writeJson(filePath, updated);
    progress.advance(relativePath);
  });
  progress.finish();
}

function parseArgs(args) {
  const options = {
    inputDir: DEFAULT_INPUT_DIR,
    model: DEFAULT_MODEL,
    concurrency: DEFAULT_CONCURRENCY,
    retries: DEFAULT_RETRIES,
    requestDelayMs: DEFAULT_REQUEST_DELAY_MS,
    force: false,
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

    if (arg === "--input-dir" || arg === "-i") {
      options.inputDir = requireValue(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--model" || arg === "-m") {
      options.model = requireValue(arg, value);
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

    if (arg === "--request-delay-ms" || arg === "--delay-ms") {
      options.requestDelayMs = parsePositiveInteger(arg, value);
      index += 1;
      continue;
    }

    if (arg === "--force") {
      options.force = true;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  npm run score-headlines
  npm run score-headlines -- --session 0
  npm run score-headlines -- --max-sessions 20 --concurrency 100
  npm run score-headlines -- --force --session 0

Required:
  Set GEMINI_API_KEY in .env or the process environment.

Options:
  --input-dir, -i     Directory containing session-*.json files (default: ${DEFAULT_INPUT_DIR})
  --model, -m         Gemini model to use (default: ${DEFAULT_MODEL})
  --session, -s       Score only a specific session; repeatable
  --max-sessions      Limit how many session files to process
  --concurrency, -c   Parallel Gemini requests (default: ${DEFAULT_CONCURRENCY})
  --retries           Retries per session on API failure (default: ${DEFAULT_RETRIES})
  --request-delay-ms  Minimum delay between Gemini request starts (default: ${DEFAULT_REQUEST_DELAY_MS})
  --force             Rescore files even if they already contain sentiment data`);
}

async function resolveSessionFiles(inputDir, options) {
  const entries = await readdir(inputDir, { withFileTypes: true });
  let files = entries
    .filter((entry) => entry.isFile() && /^session-\d+\.json$/u.test(entry.name))
    .map((entry) => path.join(inputDir, entry.name))
    .sort((left, right) => extractSessionId(left).localeCompare(extractSessionId(right), undefined, { numeric: true }));

  if (options.sessionIds.length > 0) {
    const selected = new Set(options.sessionIds);
    files = files.filter((filePath) => selected.has(extractSessionId(filePath)));
  }

  if (options.maxSessions !== null) {
    files = files.slice(0, options.maxSessions);
  }

  return files;
}

async function filterAlreadyScoredFiles(files, force) {
  if (force) {
    return files;
  }

  const filtered = [];

  for (const filePath of files) {
    const sessionData = JSON.parse(await readFile(filePath, "utf8"));
    if (!isAlreadyScored(sessionData)) {
      filtered.push(filePath);
    }
  }

  return filtered;
}

async function scoreSessionFile({ apiKey, model, retries, scheduler, sessionData }) {
  if (!Array.isArray(sessionData.headlines) || sessionData.headlines.length === 0) {
    throw new Error(`Session ${sessionData.session ?? "unknown"} has no headlines array.`);
  }

  const scores = await callGeminiWithRetry({
    apiKey,
    model,
    retries,
    scheduler,
    prompt: buildPrompt(sessionData),
  });

  const parsed = JSON.parse(scores);
  const scoreMap = new Map(parsed.headlineScores.map((item) => [item.headlineId, item]));

  const headlines = sessionData.headlines.map((headline) => {
    const score = scoreMap.get(headline.headlineId);
    if (!score) {
      throw new Error(
        `Gemini response for session ${sessionData.session} is missing score for ${headline.headlineId}.`,
      );
    }

    return {
      ...headline,
      sentiment: {
        reasoning: score.reasoning,
        longTerm: score.longTerm,
        shortTerm: score.shortTerm,
        longTermUncertainty: score.longTermUncertainty,
        shortTermUncertainty: score.shortTermUncertainty,
        impactType: score.impactType,
        evidenceStrength: score.evidenceStrength,
        durability: score.durability,
      },
    };
  });

  return {
    ...sessionData,
    scoring: {
      model,
      scoredAt: new Date().toISOString(),
      scale: {
        longTerm: "integer from -10 to 10",
        shortTerm: "integer from -10 to 10",
        longTermUncertainty: "integer from 0 to 10",
        shortTermUncertainty: "integer from 0 to 10",
      },
      methodology:
        "Gemini scores only the effect implied by the headline text itself, then separates immediate reaction versus later carry-through within the same 100-day session.",
    },
    headlines,
  };
}

function buildPrompt(sessionData) {
  const lines = sessionData.headlines
    .map(
      (headline) =>
        `- ${headline.headlineId} | company ${headline.companyId} (${headline.companyName}) | bar ${headline.barIx} | ${headline.headline}`,
    )
    .join("\n");

  return `Score each headline independently for market impact within a 100-day session.

Use two explicit horizons for every headline:
- short term = the next 1 to 10 days after the headline
- long term = roughly days 30 to 100 after the headline, meaning whether the effect still matters well after the initial reaction

Scales:
- shortTerm: integer from -10 to 10. Negative means harmful in the early part of the 100-day session, positive means beneficial in the early part of the 100-day session.
- longTerm: integer from -10 to 10. Negative means harmful in the later part of the same 100-day session, positive means beneficial in the later part of the same 100-day session.
- shortTermUncertainty: integer from 0 to 10. Higher means the effect over the next few days to few weeks in this 100-day session is unclear.
- longTermUncertainty: integer from 0 to 10. Higher means the effect over the later part of the same 100-day session is unclear.

Instructions:
- Score only the effect implied by this single headline.
- Do not add hidden assumptions about later execution risk, macro conditions, integration risk, customer churn, legal fallout, or management quality unless the headline itself mentions them.
- Uncertainty is uncertainty in your score from the headline text alone, not general uncertainty about the business world.
- Score the effect on the company named in the headline.
- Be conservative and realistic.
- Use the full range when justified.
- Return one score object for every headlineId below.
- Use this sequence for each headline:
  1. decide the impactType
  2. write one overall reasoning sentence first
  3. estimate evidenceStrength and durability
  4. then assign the numeric scores
- Keep reasoning short. One sentence only.
- shortTermUncertainty and longTermUncertainty must be judged separately.
- shortTermUncertainty depends on how clear the 1 to 10 day reaction is from the headline itself.
- longTermUncertainty depends on whether the effect persists, compounds, reverses, or fades by days 30 to 100.
- Short-term contract wins, recalls, lawsuits, earnings misses, and management changes often have different near-term and later uncertainty. Reflect that.
- The two uncertainty scores should usually differ. Set them equal only when there is a strong reason that the ambiguity is genuinely the same in both windows.
- The two impact scores should also differ whenever the headline implies a temporary shock, delayed payoff, or fading effect.
- If the headline is explicit and directional, uncertainty should be low.
- Example: "secures $320M contract" is clear positive evidence from the text itself, so both uncertainties should usually be low unless the headline includes hedging or ambiguity.
- High uncertainty is for vague headlines, speculative language, mixed signals, or cases where the horizon effect is genuinely unclear from the text.
- Put the reasoning fields before the numeric fields in each result object.
- Return strict JSON only.

Session: ${sessionData.session}
Headlines:
${lines}`;
}

async function callGeminiWithRetry({ apiKey, model, retries, scheduler, prompt }) {
  let attempt = 0;
  let lastError;

  while (attempt <= retries) {
    try {
      return await callGemini({ apiKey, model, scheduler, prompt });
    } catch (error) {
      lastError = error;
      if (attempt === retries) {
        break;
      }
      const delayMs = isRateLimitError(error) ? 1500 * 2 ** attempt : 500 * 2 ** attempt;
      await sleep(delayMs);
    }
    attempt += 1;
  }

  throw lastError;
}

async function callGemini({ apiKey, model, scheduler, prompt }) {
  const response = await scheduler(async () =>
    fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
      {
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
                text: "You assign financial-impact scores to headlines and return strict JSON matching the schema.",
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
            temperature: 0.1,
            responseMimeType: "application/json",
            responseJsonSchema: responseSchema(),
          },
        }),
      },
    ),
  );

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

function responseSchema() {
  return {
    type: "object",
    properties: {
      headlineScores: {
        type: "array",
        items: {
          type: "object",
          properties: {
            headlineId: { type: "string" },
            reasoning: { type: "string" },
            impactType: {
              type: "string",
              enum: [
                "contract_win",
                "guidance",
                "earnings",
                "expansion",
                "leadership",
                "regulatory",
                "litigation",
                "recall",
                "acquisition",
                "product_launch",
                "operations",
                "demand",
                "cost_pressure",
                "other",
              ],
            },
            evidenceStrength: { type: "integer", minimum: 0, maximum: 10 },
            durability: { type: "integer", minimum: 0, maximum: 10 },
            longTerm: { type: "integer", minimum: -10, maximum: 10 },
            shortTerm: { type: "integer", minimum: -10, maximum: 10 },
            longTermUncertainty: { type: "integer", minimum: 0, maximum: 10 },
            shortTermUncertainty: { type: "integer", minimum: 0, maximum: 10 },
          },
          required: [
            "headlineId",
            "reasoning",
            "impactType",
            "evidenceStrength",
            "durability",
            "longTerm",
            "shortTerm",
            "longTermUncertainty",
            "shortTermUncertainty",
          ],
          additionalProperties: false,
        },
      },
    },
    required: ["headlineScores"],
    additionalProperties: false,
  };
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

function createRequestScheduler(delayMs) {
  let nextAvailableAt = 0;

  return async function schedule(task) {
    const now = Date.now();
    const waitMs = Math.max(0, nextAvailableAt - now);
    nextAvailableAt = Math.max(now, nextAvailableAt) + delayMs;

    if (waitMs > 0) {
      await sleep(waitMs);
    }

    return task();
  };
}

function createProgressBar(total) {
  let completed = 0;

  function render(lastFile = "") {
    const width = 24;
    const ratio = total === 0 ? 1 : completed / total;
    const filled = Math.round(width * ratio);
    const bar = `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
    const percent = String(Math.round(ratio * 100)).padStart(3, " ");
    const suffix = lastFile ? ` ${lastFile}` : "";
    process.stdout.write(`\r[${bar}] ${completed}/${total} ${percent}%${suffix}`);
  }

  render();

  return {
    advance(lastFile) {
      completed += 1;
      render(lastFile);
    },
    finish() {
      process.stdout.write("\n");
    },
  };
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

function isRateLimitError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return /429|rate limit|quota/i.test(message);
}

function extractSessionId(filePath) {
  return path.basename(filePath).replace(/^session-(\d+)\.json$/u, "$1");
}

function isAlreadyScored(sessionData) {
  if (!Array.isArray(sessionData?.headlines) || sessionData.headlines.length === 0) {
    return false;
  }

  return sessionData.headlines.every(
    (headline) =>
      headline &&
      typeof headline === "object" &&
      headline.sentiment &&
      typeof headline.sentiment === "object" &&
      Number.isInteger(headline.sentiment.shortTerm) &&
      Number.isInteger(headline.sentiment.longTerm) &&
      Number.isInteger(headline.sentiment.shortTermUncertainty) &&
      Number.isInteger(headline.sentiment.longTermUncertainty),
  );
}

async function writeJson(filePath, payload) {
  await writeFile(filePath, JSON.stringify(payload, null, 2));
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
