# Polymarket FAA attribution forecast — entry

Forecasts, minute by minute and 4 hours ahead, the Yes price of every
market of Polymarket's FAA outage attribution family (Iran / foreign actor /
hack / AI sandbox escape) as five quantiles, at each of twelve checkpoints.

## The program

`main.py` (Python 3.13, standard library only, deterministic, no network):
reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`), writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

- **Median = last price** (persistence). Adding drift from recent returns
  hurt on every practice variant tried.
- **Spread**: winsorized std of the market's 1-minute returns over the
  last 120 minutes (moves clipped at ±0.02 so one jump cannot inflate the
  width for hours), floored at 0.0005, capped at 0.008.
- **Width grows as h^0.4**, not sqrt(h): prices are sticky over 4 hours.
- **Outer quantiles (0.05 / 0.95) widened ×1.15** for jump risk.

## Validation (practice, local scorer)

Local scorer (`score.py`; answers from the next case file, p05's from
Polymarket history) reproduces the house baseline's stated practice loss
**0.363589** exactly, so the scorer is trustworthy.

| entry | mean pinball loss (practice) |
|---|---|
| house baseline (fixed sigma 0.0025, sqrt h) | 0.363589 |
| **this entry** | **0.270461 (−25.6%)** |

Beats the baseline on **all six** practice cases (p00 0.469 vs 0.536,
p01 0.313 vs 0.380, p02 0.338 vs 0.408, p03 0.244 vs 0.369, p04 0.200 vs
0.299, p05 0.122 vs 0.238). Runtime ~0.14 s/case (cap 8 s). Outputs
valid: 240 rows × 5 monotone quantiles in [0, 1], ~270 KB (< 1 MiB).

Caveat (from the rules): the practice is the NVIDIA event, not the FAA
family, so the practice score is no guide to the final. The final markets
are young (~16 h old at the deadline) and the estimator only looks back
120 minutes, so it adapts to young markets the same way.

## Files

- `main.py` — the entry (the only file the submission needs, plus a
  LICENSE and `.makefun/owner.json` added by the parent at submit time).
- `NOTES.md` — tuning log.
- `score.py`, `sweep*.py`, `fetch_p05.py`, `answers-p05.json`,
  `out-baseline/`, `out-v1/` — local dev tooling and validation outputs.
