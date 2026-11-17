# Ratio-Anchored Ensemble — GTA VI launch-day Twitch CCU forecaster

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rebahan149/gta6-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rebahan149/gta6-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the make.fun bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast a game's launch-day Twitch average viewers for hours 2–24 of the
release day (UTC), as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per
hour, given only the hour-1 print and a library of comparable launches.
Scoring is mean pinball loss on log viewers, so the whole game is a
well-calibrated predictive distribution per hour — not a point guess.

`main.py` is the entire entrypoint: Python 3 standard library only, no
network, fully deterministic (same input → byte-identical output).

## Method

**Comparable-ratio ensemble.** Every comparable launch in the case file is
reduced to its release-day curve, and each hour is expressed as a log-ratio
to that launch's own hour 1. The forecast for an hour is a weighted
distribution of those ratios across comparable launches, exponentiated and
scaled by the target's known hour-1 print.

**Membership filters** (the part that makes ratios meaningful):

- Drop the kit's `early-access-contaminated` and `weak` tier launches —
  their day-one curves measure a staggered early-access crowd.
- Drop single global-unlock launches via a *maturity* filter
  (hour 1 < 5% of the day's peak): with one worldwide unlock time, hour 1
  prints near zero and ratios to it explode; the rolling-midnight family
  GTA VI belongs to sits far above that line.
- Drop the target's own pre-release category history.

**Member weights** reward resemblance to the target: cosine similarity of
the demeaned 24 h pre-release log ramp (squared), an early-ramp kernel on
`ln(hour1 / mean viewers 12–7 h before the day)`, a 730-day recency
half-life, and a 3× bonus for console-only launches when the target is
console-only (parsed from the case notes).

**Distribution per hour.** Weighted empirical quantiles of the log-ratios,
blended 50/50 with a lognormal fit (median + IQR-based sigma), widened by a
calibrated ×1.2 around the median, with a per-hour sigma floor of 0.04.
If no comparable survives the filters, the entry falls back to the
persistence baseline (hour-1 level, lognormal spread growing as √h).

## Calibration

Every constant above was chosen by leave-one-out cross-validation over the
comparable library (each launch forecast from the others) and checked on
the four practice cases — never fitted to the practice answers directly,
because the prize grade is one unseen launch day.

Practice results with the exact competition scorer (mean pinball loss on
log viewers, lower is better):

| model | practice loss |
| --- | --- |
| persistence baseline (kit starter / house baseline) | 0.3871 |
| flat-weight ratio ensemble, empirical quantiles | 0.2645 |
| tuned model (lab) | 0.2438 |
| this entry, run through the grader's file interface | **0.2407** |

Leave-one-out over the rolling-midnight family: 0.3143. The fitted day
shape for a rolling console launch: dip to ≈0.7× of hour 1 in hours 3–4,
recovery by hour 7, then a long ramp to a median peak of ≈3.0× hour 1 at
hour 19 (90% interval at the peak hour roughly 1.0×–9.5×), easing to
≈2.5× by hour 24.

## Layout

- `main.py` — the forecaster (entrypoint; reads `$MAKEFUN_INPUT_DIR`,
  writes `$MAKEFUN_OUTPUT_DIR`, one `<id>.out` per `<id>.in` case).
- `.makefun/owner.json` — entrant wallet.
- `LICENSE` — MIT.
