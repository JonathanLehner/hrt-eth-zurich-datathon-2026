import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { asyncBufferFromFile, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";

const DEFAULT_FILE = "data/headlines_seen_train.parquet";
const DEFAULT_OUTPUT_DIR = "analysis/company-ids";
const TITLE_STOPWORDS = new Set([
  "Chief",
  "CEO",
  "CFO",
  "COO",
  "CTO",
  "President",
  "Chairman",
  "Director",
  "Officer",
  "Head",
]);

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const filePath = path.resolve(process.cwd(), options.file);
  const outputDir = path.resolve(process.cwd(), options.outputDir);
  const sessions = await loadSessions(filePath);
  const selectedSessions = filterSessions(sessions, options);

  if (selectedSessions.length === 0) {
    throw new Error("No sessions matched the provided filters.");
  }

  await mkdir(outputDir, { recursive: true });

  console.log(
    `Assigning company IDs for ${selectedSessions.length} session(s) from ${path.relative(process.cwd(), filePath)}`,
  );

  const index = [];

  for (const [sessionId, headlines] of selectedSessions) {
    const sessionOutput = buildSessionOutput(sessionId, headlines);
    const sessionFile = path.join(outputDir, `session-${sessionId}.json`);
    await writeJson(sessionFile, sessionOutput);
    index.push({
      session: sessionId,
      file: path.relative(process.cwd(), sessionFile),
      companyCount: sessionOutput.companies.length,
      headlineCount: sessionOutput.headlines.length,
    });
    console.log(
      `Saved session ${sessionId}: ${sessionOutput.companies.length} companies, ${sessionOutput.headlines.length} headlines`,
    );
  }

  const summaryFile = path.join(outputDir, "index.json");
  await writeJson(summaryFile, {
    generatedAt: new Date().toISOString(),
    sourceFile: path.relative(process.cwd(), filePath),
    sessionCount: index.length,
    sessions: index,
  });

  console.log(`Saved index to ${path.relative(process.cwd(), summaryFile)}`);
}

function parseArgs(args) {
  const options = {
    file: DEFAULT_FILE,
    outputDir: DEFAULT_OUTPUT_DIR,
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

    if (arg === "--output-dir" || arg === "-o") {
      options.outputDir = requireValue(arg, value);
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

    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  npm run assign-company-ids
  npm run assign-company-ids -- --session 0
  npm run assign-company-ids -- --file data/headlines_seen_public_test.parquet --max-sessions 25

Options:
  --file, -f         Headlines parquet file to analyze (default: ${DEFAULT_FILE})
  --output-dir, -o   Directory for per-session JSON files (default: ${DEFAULT_OUTPUT_DIR})
  --session, -s      Analyze only a specific session; repeatable
  --max-sessions     Limit how many sessions to process`);
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
      session: sessionId,
      barIx: toComparableNumber(row.bar_ix),
      text: row.headline,
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

function buildSessionOutput(sessionId, headlines) {
  const companies = new Map();
  const annotatedHeadlines = headlines.map((headline, index) => {
    const companyName = extractCompanyName(headline.text);
    const company = upsertCompany(companies, companyName, headline, index);

    return {
      headlineId: `H${String(index + 1).padStart(3, "0")}`,
      barIx: headline.barIx,
      companyId: company.companyId,
      companyName: company.name,
      headline: headline.text,
    };
  });

  const companyList = [...companies.values()]
    .map((company) => ({
      companyId: company.companyId,
      name: company.name,
      headlineCount: company.headlineCount,
      firstBarIx: company.firstBarIx,
      headlineIds: company.headlineIds,
      sampleHeadlines: company.sampleHeadlines,
    }))
    .sort((left, right) => left.companyId - right.companyId);

  return {
    session: sessionId,
    companyCount: companyList.length,
    headlineCount: annotatedHeadlines.length,
    companies: companyList,
    headlines: annotatedHeadlines,
  };
}

function upsertCompany(companies, companyName, headline, index) {
  if (!companies.has(companyName)) {
    const companyId = companies.size;
    companies.set(companyName, {
      companyId,
      name: companyName,
      headlineCount: 0,
      firstBarIx: headline.barIx,
      headlineIds: [],
      sampleHeadlines: [],
    });
  }

  const company = companies.get(companyName);
  company.headlineCount += 1;
  company.firstBarIx = Math.min(company.firstBarIx, headline.barIx);
  company.headlineIds.push(`H${String(index + 1).padStart(3, "0")}`);

  if (company.sampleHeadlines.length < 3) {
    company.sampleHeadlines.push(headline.text);
  }

  return company;
}

function extractCompanyName(headline) {
  const tokens = headline.split(/\s+/u).filter(Boolean);
  const companyTokens = [];

  for (const token of tokens) {
    const cleaned = token.replace(/^[("'`]+|[,.:;!?)"'`]+$/gu, "");

    if (!cleaned) {
      break;
    }

    if (TITLE_STOPWORDS.has(cleaned)) {
      break;
    }

    if (!looksLikeCompanyToken(cleaned)) {
      break;
    }

    companyTokens.push(cleaned);
  }

  if (companyTokens.length >= 2) {
    return companyTokens.join(" ");
  }

  if (companyTokens.length === 1) {
    return companyTokens[0];
  }

  return headline;
}

function looksLikeCompanyToken(token) {
  if (token === "&" || token === "and") {
    return true;
  }

  return /^[A-Z][A-Za-z0-9'&.-]*$/u.test(token);
}

function toComparableNumber(value) {
  if (typeof value === "bigint") {
    return Number(value);
  }
  return Number(value);
}

async function writeJson(filePath, payload) {
  await writeFile(filePath, JSON.stringify(payload, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
