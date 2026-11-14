#!/usr/bin/env python3
# polymarket-faa-attribution-02 entry (Python 3.13, standard library only):
# forecast each market's Yes price at every minute of the next 4 hours as
# five quantiles, from the minute prices in the case file only. No network,
# no wall clock, deterministic.
#
# Model (tuned on the kit's six practice cases, NVIDIA event):
#   - median = the market's last price (persistence; drift hurts on practice).
#   - spread: a winsorized standard deviation of the market's 1-minute
#     returns over the last 120 minutes (clipping moves at +/-0.02 so one
#     jump cannot blow up the width), floored at 0.0005 and capped at 0.008
#     price units per minute.
#   - width grows as h**0.4, not sqrt(h): Polymarket prices are sticky and
#     mean-reverting over 4 hours, so the move distribution saturates
#     (alpha 0.4 beat 0.5 by ~0.005 and 0.6 by ~0.04 on practice).
#   - the outer quantiles (0.05 / 0.95) are widened x1.15 relative to the
#     normal quantiles: jumps are rare but real, and the outer levels pay
#     for them.
#   - quantiles clamped to [0, 1], rounded to 6 decimals, monotone.
#
# Practice (local scorer, answers from the next case file + Polymarket
# history for p05; scorer reproduces the house baseline's stated
# 0.363589 exactly): this entry 0.270461 vs house baseline 0.363589
# (-25.6%). The house model forecast is a one-shot superbot answer the
# house replays with the same q = x + drift*h/60 + z*sigma*sqrt(h) form;
# this entry's adaptive per-market sigma + sub-diffusive width is a
# different, tighter family and beats the baseline by a wide margin.

import json
import math
import os
import statistics
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level, to 6 decimals.
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
# Outer quantiles are widened x1.15 (jump risk); inner levels stay normal.
Z_MULT = (1.15, 1.0, 1.0, 1.0, 1.15)

# Winsorized-sigma parameters (see the module docstring).
LOOKBACK_MIN = 120
CLIP_MOVE = 0.02
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.008
# Width grows as h ** ALPHA.
ALPHA = 0.4


def _norm_ppf(p):
    """Acklam's rational approximation to the standard normal quantile.

    Only used when a case asks for quantile levels other than the five
    standard ones (the contest fixes them, so this is a fallback).
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _sigma(prices):
    """Per-minute spread of one market: the winsorized std of its 1-minute
    returns over the last LOOKBACK_MIN minutes, floored and capped.

    Winsorizing at +/-CLIP_MOVE keeps a single jump (the FAA markets moved
    0.2+ in a minute at creation; the NVIDIA practice has jumps to 0.48)
    from inflating the width for hours afterwards. A market with fewer than
    5 returns in the window falls back to the house baseline's 0.0025.
    """
    n = len(prices)
    if n < 2:
        return 0.0025
    lo = max(1, n - LOOKBACK_MIN)
    rets = [prices[i] - prices[i - 1] for i in range(lo, n)]
    if len(rets) < 5:
        return 0.0025
    clipped = [max(-CLIP_MOVE, min(CLIP_MOVE, r)) for r in rets]
    s = statistics.pstdev(clipped)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, s))


def _price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def forecast_market(prices, horizon, quantiles):
    """240 rows (horizon rows) of five quantiles for one market.

    Row h-1 is the forecast for minute at+h: median = last price, spread =
    sigma * h**ALPHA, outer quantiles widened by Z_MULT.
    """
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    sig = _sigma(prices)
    if tuple(quantiles) == QUANTILES:
        zs = [z * m for z, m in zip(Z, Z_MULT)]
    else:
        zs = [_norm_ppf(t) for t in quantiles]
        # Widen the two extreme levels, as with the standard five.
        zs[0] *= 1.15
        zs[-1] *= 1.15
    rows = []
    for h in range(1, horizon + 1):
        w = sig * (h ** ALPHA)
        row = [_price6(x + z * w) for z in zs]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        print(f"no .in files in {INPUT_DIR}", file=sys.stderr)
        return 1
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        quantiles = list(case["quantiles"])
        forecasts = []
        for m in case["markets"]:
            q = forecast_market(list(m["prices"]), horizon, quantiles)
            forecasts.append({"id": m["id"], "q": q})
        out_path = os.path.join(OUTPUT_DIR, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
