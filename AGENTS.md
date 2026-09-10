# derivatives-bt-engine

## Python

- Use the project virtual environment at `.venv`.
- Run Python using `.venv/bin/python`.
- Use Polars rather than pandas for project data processing.
- Use the project's existing data-format conventions.
- Do not install packages globally.

## Testing and execution

- Project CLI executables under `.venv/bin/` may be used for testing.
- Backtests may be computationally expensive; use narrow symbol/year ranges for validation before running large tests.
- Preserve existing backtest results unless explicitly asked to regenerate or overwrite them.

## Data

- Financial data may exist outside the repository under `/home/dev/data/fin`.
- Database resources may exist under `/home/dev/fin/db`.
- Do not modify source market data unless explicitly requested.

## Interactive Brokers

- Local IB Gateway/TWS ports may include 7496, 7497, 4001, and 4002.
- `nc` may be used to test whether these local ports are reachable.

## Repository

- Treat this repository as private.
- Treat `.env` files and credentials as sensitive.

## Workflow

- After each completed task, commit the task changes and push the commit to the configured remote.
- Include 1–3 sentences of context in each commit message describing what changed and why.
- Keep unrelated user changes out of task commits.
