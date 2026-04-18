import { readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  asyncBufferFromFile,
  parquetMetadataAsync,
  parquetReadObjects,
  parquetSchema,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

const DEFAULT_ROWS = 5;
const DATA_DIR = path.resolve(process.cwd(), "data");

async function main() {
  const { rows, targets } = parseArgs(process.argv.slice(2));
  const parquetFiles = await resolveParquetFiles(targets);

  if (parquetFiles.length === 0) {
    console.error("No .parquet files found to display.");
    process.exitCode = 1;
    return;
  }

  console.log(`Reading ${parquetFiles.length} parquet file(s) from ${DATA_DIR}`);

  for (const filePath of parquetFiles) {
    await displayParquetFile(filePath, rows);
  }
}

function parseArgs(args) {
  const targets = [];
  let rows = DEFAULT_ROWS;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === "--rows" || arg === "-r") {
      const value = args[index + 1];
      if (!value || Number.isNaN(Number(value))) {
        throw new Error("Expected a numeric value after --rows.");
      }
      rows = Number(value);
      index += 1;
      continue;
    }

    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    targets.push(arg);
  }

  if (!Number.isInteger(rows) || rows < 1) {
    throw new Error("--rows must be a positive integer.");
  }

  return { rows, targets };
}

function printHelp() {
  console.log(`Usage:
  npm run show-data
  npm run show-data -- --rows 3
  npm run show-data -- data/headlines_seen_train.parquet
  npm run show-data -- bars_seen_train.parquet headlines_seen_train.parquet --rows 2`);
}

async function resolveParquetFiles(targets) {
  if (targets.length === 0) {
    const entries = await readdir(DATA_DIR, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".parquet"))
      .map((entry) => path.join(DATA_DIR, entry.name))
      .sort();
  }

  return targets
    .map((target) => {
      const resolved = path.resolve(process.cwd(), target);
      if (resolved.endsWith(".parquet")) {
        return resolved;
      }
      return path.join(DATA_DIR, target);
    })
    .sort();
}

async function displayParquetFile(filePath, rows) {
  const file = await asyncBufferFromFile(filePath);
  const metadata = await parquetMetadataAsync(file);
  const schema = parquetSchema(metadata);
  const sampleRows = await parquetReadObjects({
    file,
    compressors,
    rowStart: 0,
    rowEnd: rows,
  });

  const columns = (schema.children || []).map((child) => child.element.name);

  console.log("");
  console.log(`File: ${path.relative(process.cwd(), filePath)}`);
  console.log(`Rows: ${safeNumber(metadata.num_rows)}`);
  console.log(`Columns (${columns.length}): ${columns.join(", ")}`);
  console.log(`Sample (${sampleRows.length} row(s)):`);
  console.log(stringifySample(sampleRows));
}

function safeNumber(value) {
  if (typeof value === "bigint") {
    return value.toString();
  }
  return String(value);
}

function stringifySample(rows) {
  return JSON.stringify(
    rows,
    (_, value) => {
      if (typeof value === "bigint") {
        return value.toString();
      }
      if (value instanceof Uint8Array) {
        return `Uint8Array(${value.length})`;
      }
      return value;
    },
    2,
  );
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
