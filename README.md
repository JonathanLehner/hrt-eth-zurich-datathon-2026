# hrt-eth-zurich-datathon-2026

## Setup

Install dependencies:

```bash
npm install
```

For Gemini-based scripts, create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Commands

### Show parquet data

Preview the parquet files in `./data`, including row counts, columns, and sample rows.

```bash
npm run show-data
npm run show-data -- --rows 3
npm run show-data -- data/headlines_seen_train.parquet --rows 1
```

### Analyze headlines with Gemini

Reads a headlines parquet file, groups rows by `session`, and asks Gemini which company is most likely the real tracked company in each session.

Default input:

```bash
data/headlines_seen_train.parquet
```

Examples:

```bash
npm run analyze-headlines -- --session 0
npm run analyze-headlines -- --max-sessions 25
npm run analyze-headlines -- --file data/headlines_seen_public_test.parquet --concurrency 2
```

Output:

- Default output file is `analysis/<input-file-name>.gemini.json`
- Example: `analysis/headlines_seen_train.gemini.json`

Useful flags:

- `--file` or `-f`: parquet file to analyze
- `--session` or `-s`: analyze only one session, repeatable
- `--max-sessions`: limit number of sessions
- `--model` or `-m`: Gemini model, default is `gemini-2.5-flash`
- `--concurrency` or `-c`: parallel Gemini requests
- `--retries`: retries per session
- `--output` or `-o`: custom output file

### Assign company IDs

Reads a headlines parquet file, groups headlines by `session`, extracts the company name from each headline, assigns numeric session-local company IDs from `0` to `X`, and writes one JSON file per session.

Examples:

```bash
npm run assign-company-ids
npm run assign-company-ids -- --session 0
npm run assign-company-ids -- --file data/headlines_seen_public_test.parquet --max-sessions 25
```

Output:

- Default output directory is `analysis/company-ids`
- Per-session files are saved as `analysis/company-ids/session-<sessionId>.json`
- A summary index is saved as `analysis/company-ids/index.json`

Each session file contains:

- top-level `session`
- `companies`: company list with numeric `companyId`
- `headlines`: every headline with its assigned `companyId`

Useful flags:

- `--file` or `-f`: parquet file to analyze
- `--output-dir` or `-o`: output directory
- `--session` or `-s`: process only one session, repeatable
- `--max-sessions`: limit number of sessions

### Score headlines with Gemini

Reads existing `analysis/company-ids/session-*.json` files, scores every headline, and writes the results back into the same session files.

For each headline it adds:

- `reasoning`
- `longTerm`: integer from `-10` to `10`
- `shortTerm`: integer from `-10` to `10`
- `longTermUncertainty`: integer from `0` to `10`
- `shortTermUncertainty`: integer from `0` to `10`

Examples:

```bash
npm run score-headlines
npm run score-headlines -- --session 0
npm run score-headlines -- --max-sessions 20 --concurrency 2
```

Input:

- Default input directory is `analysis/company-ids`
- The command expects files named `session-*.json`

Useful flags:

- `--input-dir` or `-i`: directory containing session files
- `--session` or `-s`: score only one session, repeatable
- `--max-sessions`: limit number of sessions
- `--model` or `-m`: Gemini model, default is `gemini-2.5-flash`
- `--concurrency` or `-c`: parallel Gemini requests
- `--retries`: retries per session

## Typical workflow

If you want the full flow on the training headlines:

```bash
npm install
npm run show-data -- data/headlines_seen_train.parquet --rows 2
npm run assign-company-ids -- --file data/headlines_seen_train.parquet
npm run score-headlines -- --session 0
npm run analyze-headlines -- --session 0
```

If you want to process a different headlines file:

```bash
npm run assign-company-ids -- --file data/headlines_seen_public_test.parquet --output-dir analysis/company-ids-public
npm run score-headlines -- --input-dir analysis/company-ids-public --session 1000
```

## Notes

- `assign-company-ids` is deterministic and does not call Gemini.
- `analyze-headlines` and `score-headlines` require `GEMINI_API_KEY`.
- In restricted sandboxes, Gemini requests may fail if outbound network access is blocked.
