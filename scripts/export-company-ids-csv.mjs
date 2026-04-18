import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const DEFAULT_INPUT_DIR = "analysis/company-ids";
const DEFAULT_OUTPUT = "analysis/company-ids.csv";
const HEADER = [
  "session",
  "barIx",
  "companyId",
  "longTerm",
  "shortTerm",
  "longTermUncertainty",
  "shortTermUncertainty",
];

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(process.cwd(), options.inputDir);
  const outputPath = path.resolve(process.cwd(), options.output);
  const sessionFiles = await listSessionFiles(inputDir);

  const rows = [];

  for (const filePath of sessionFiles) {
    const sessionData = JSON.parse(await readFile(filePath, "utf8"));
    if (!Array.isArray(sessionData.headlines)) {
      continue;
    }

    for (const headline of sessionData.headlines) {
      const sentiment = headline?.sentiment ?? {};
      rows.push([
        sessionData.session ?? "",
        headline?.barIx ?? "",
        headline?.companyId ?? "",
        sentiment.longTerm ?? "",
        sentiment.shortTerm ?? "",
        sentiment.longTermUncertainty ?? "",
        sentiment.shortTermUncertainty ?? "",
      ]);
    }
  }

  const csv = [HEADER, ...rows].map(toCsvLine).join("\n");
  await writeFile(outputPath, `${csv}\n`);
  console.log(`Wrote ${rows.length} row(s) to ${path.relative(process.cwd(), outputPath)}`);
}

function parseArgs(args) {
  const options = {
    inputDir: DEFAULT_INPUT_DIR,
    output: DEFAULT_OUTPUT,
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

    if (arg === "--output" || arg === "-o") {
      options.output = requireValue(arg, value);
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  npm run export-company-ids-csv
  npm run export-company-ids-csv -- --input-dir analysis/company-ids
  npm run export-company-ids-csv -- --output analysis/company-ids.csv

Options:
  --input-dir, -i   Directory containing session-*.json files (default: ${DEFAULT_INPUT_DIR})
  --output, -o      Output CSV file (default: ${DEFAULT_OUTPUT})`);
}

function requireValue(flag, value) {
  if (!value || value.startsWith("--")) {
    throw new Error(`Expected a value after ${flag}.`);
  }
  return value;
}

async function listSessionFiles(inputDir) {
  const entries = await readdir(inputDir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && /^session-\d+\.json$/u.test(entry.name))
    .map((entry) => path.join(inputDir, entry.name))
    .sort((left, right) =>
      extractSessionId(left).localeCompare(extractSessionId(right), undefined, { numeric: true }),
    );
}

function extractSessionId(filePath) {
  return path.basename(filePath).replace(/^session-(\d+)\.json$/u, "$1");
}

function toCsvLine(values) {
  return values
    .map((value) => {
      const stringValue = String(value ?? "");
      if (/[",\n]/u.test(stringValue)) {
        return `"${stringValue.replace(/"/gu, "\"\"")}"`;
      }
      return stringValue;
    })
    .join(",");
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
