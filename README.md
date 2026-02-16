# Vectorized Multi-Alpha Backtest

This project builds and runs 3 equity alphas on S&P 500 data, then compares performance and runtime before vs after vectorization.

## What This Project Does

- Pulls S&P 500 ticker symbols from Wikipedia.
- Downloads daily OHLCV data with `yfinance`.
- Computes strategy-specific alpha signals:
  - `Alpha1`: cross-sectional pressure/ranking signal
  - `Alpha2`: open-vs-close mean-reversion style signal
  - `Alpha3`: trend composite from moving-average crossovers
- Runs a vectorized backtest engine with volatility targeting.
- Prints final capital for each strategy.

## Project Structure

- `src/main.py`: data load + run entrypoint
- `src/utils.py`: core backtest engine, portfolio math, helpers
- `src/alpha1.py`: Alpha1 signal definition
- `src/alpha2.py`: Alpha2 signal definition
- `src/alpha3.py`: Alpha3 signal definition
- `run.sh`: one-command runner
- `stats`: benchmark notes (runtime/capital snapshots)

## Requirements

Install dependencies:

```bash
pip install -r requirement.txt
```

## How To Run

From project root (`QT101`):

```bash
./run.sh
```

Equivalent command:

```bash
python3 -m src.main
```

## Current Runtime / Result Snapshot

From `stats`:

### Before vectorization (200 tickers, alpha1/2/3)

- `@timeme: run_simulation took 29.2004611492157 seconds` -> final capital `16042.507221192447`
- `@timeme: run_simulation took 31.06265115737915 seconds` -> final capital `12082.024138006269`
- `@timeme: run_simulation took 30.710301160812378 seconds` -> final capital `40813.95841306449`

### After vectorization (200 tickers, alpha1/2/3)

- `@timeme: run_simulation took 2.343602180480957 seconds` -> final capital `16042.507221192447`
- `@timeme: run_simulation took 2.1614718437194824 seconds` -> final capital `12082.024138006269`
- `@timeme: run_simulation took 2.172659158706665 seconds` -> final capital `40813.95841306449`

## Notes

- Data is cached in `dataset.obj` once downloaded.
- Backtests use UTC timestamps and daily bars.
- Warning suppression for `FutureWarning` and `RuntimeWarning` is currently enabled in `src/main.py`.
