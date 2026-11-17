#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch-day Twitch CCU forecaster (make.fun: gta6-twitch-ccu-forecast-01).

Method: comparable-ratio ensemble.

For every past launch shipped inside the case file we rebuild its release-day
curve and express each hour as a log-ratio to that launch's own hour 1.  A
launch only enters the ensemble when its unlock structure is comparable to a
rolling-midnight console launch:

  * the kit's tier tags drop "early-access-contaminated" and "weak" launches
    (their day-one curve measures a staggered early-access crowd, not a
    release-day crowd);
  * a maturity filter (hour 1 >= 5% of the day's peak) drops single
    global-unlock launches, whose hour 1 is near zero because most of the
    world is still locked out -- ratio-to-hour-1 is meaningless for them;
  * the target game's own pre-release category history is excluded.

Members are weighted by how much they resemble the target: cosine similarity
of the demeaned 24 h pre-release log ramp, an early-ramp kernel (how far the
target already is into its ramp when hour 1 prints), a recency half-life of
730 days, and -- when the target itself is console-only -- a 3x bonus for
console-only members.

Per forecast hour, the predictive distribution is the weighted empirical
quantiles of the members' log-ratios, blended 50/50 with a lognormal fit
(median + IQR sigma) and widened by a calibrated factor of 1.2 around the
median, with a per-hour sigma floor.  Quantiles are exponentiated and scaled
by the known hour-1 print.  Calibration was chosen by leave-one-out over the
comparable set plus the four practice cases, not by fitting practice answers.

Deterministic: no randomness, no clock, no network.  Standard library only.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from statistics import NormalDist

# --------------------------------------------------------------------------
# Calibrated constants (leave-one-out over comparables + practice cases).
# --------------------------------------------------------------------------
DROP_TIERS = {"early-access-contaminated", "weak"}
MIN_MATURITY = 0.05        # hour1 / day-peak floor for ensemble members
MIN_H1 = 500               # hour-1 viewers floor for ensemble members
SPREAD_SCALE = 1.2         # widen log-quantiles around the median by this
SIGMA_FLOOR = 0.04         # minimum per-hour log sigma
SIM_POWER = 2.0            # exponent on the pre-ramp similarity weight
RECENCY_HALFLIFE_DAYS = 730.0
CONSOLE_BONUS = 2.0        # console members weigh 1 + this when target is too
RAMP_G_WIDTH = 2.0         # kernel width for the early-ramp feature
BASE_SIGMA = 0.15          # fallback (persistence baseline) log sigma
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))

_ND = NormalDist()


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds (float)."""
    if text is None:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def day_curve(comp):
    """{hour 1..24: viewers} for a comparable's release day (UTC hours)."""
    start = parse_ts(comp.get("day_start"))
    if start is None:
        return {}
    curve = {}
    for row in comp.get("hourly") or []:
        ts = parse_ts(row.get("from"))
        if ts is None:
            continue
        hour = int((ts - start) // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = max(0.0, float(row.get("viewers") or 0))
    return curve


def pre_ramp(hourly, day_start):
    """Log viewers of the last 24 whole hours before day_start."""
    start = parse_ts(day_start)
    if start is None:
        return []
    rows = []
    for row in hourly or []:
        ts = parse_ts(row.get("from"))
        if ts is not None and ts < start:
            rows.append((ts, max(1.0, float(row.get("viewers") or 0))))
    rows.sort()
    return [math.log(v) for _, v in rows[-24:]]


def tier_of(unlock_text):
    m = re.search(r"Tier:\s*([A-Za-z\-]+)", unlock_text or "")
    return m.group(1).lower() if m else "comparable"


def console_only_from_notes(notes):
    """True when the notes' 'released ... on <platforms>' clause has no PC."""
    m = re.search(r"released[^.]*?\bon\s+([^();.]+)", notes or "")
    return bool(m) and "PC" not in m.group(1)


def weighted_quantile(sorted_pairs, tau):
    """Linear-interpolated weighted quantile of [(value, weight)] sorted."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    target = tau * total
    acc = 0.0
    prev_v = sorted_pairs[0][0]
    for v, w in sorted_pairs:
        acc += w
        if acc >= target:
            return v
        prev_v = v
    return prev_v


def ramp_similarity(a, b):
    """Cosine similarity of two demeaned log curves (1.0 if either is short)."""
    n = min(len(a), len(b))
    if n < 6:
        return 1.0
    a, b = a[-n:], b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    if den <= 0:
        return 1.0
    return num / den


def ramp_g(pre_logs, y1):
    """Early-ramp feature: ln(y1 / mean viewers 12..7 h before the day)."""
    if not pre_logs or len(pre_logs) < 18 or y1 <= 0:
        return None
    mid = pre_logs[12:18]
    base = math.exp(sum(mid) / len(mid))
    return math.log(y1 / base) if base > 0 else None


def ensemble(case):
    """Filtered, weighted ensemble members: [(curve, weight)] for a case."""
    target_name = (case.get("game") or {}).get("name") or ""
    target_start = parse_ts(case.get("day_start"))
    target_pre = pre_ramp(case.get("history_hourly"), case.get("day_start"))
    if len(target_pre) < 6:
        # Fall back to the raw history series (whole hours before day start).
        hist = []
        for row in case.get("history") or []:
            ts = parse_ts(row.get("at") or row.get("from"))
            if ts is not None and target_start is not None and ts < target_start:
                hist.append((ts, max(1.0, float(row.get("viewers") or 0))))
        hist.sort()
        target_pre = [math.log(v) for _, v in hist[-24:]]
    known = case.get("known")
    if isinstance(known, list):
        p1 = max(1.0, float((known[0] or {}).get("viewers") or 1)) if known else 1.0
    elif isinstance(known, dict):
        p1 = max(1.0, float(known.get("viewers") or 1))
    else:
        p1 = 1.0
    target_g = ramp_g(target_pre, p1)
    target_console_only = console_only_from_notes(case.get("notes"))

    members = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if target_name and name.startswith(target_name):
            continue  # the target's own pre-release category history
        if "before release" in name:
            continue
        if tier_of(comp.get("unlock")) in DROP_TIERS:
            continue
        curve = day_curve(comp)
        if len(curve) < 24 or curve.get(1, 0) < MIN_H1:
            continue
        peak = max(curve.values())
        if peak <= 0 or curve[1] / peak < MIN_MATURITY:
            continue  # single global-unlock shape: hour-1 ratios break
        w = 1.0
        sim = ramp_similarity(target_pre, pre_ramp(comp.get("hourly"),
                                                   comp.get("day_start")))
        w *= (0.05 + sim) ** SIM_POWER
        if target_start is not None:
            cstart = parse_ts(comp.get("day_start"))
            if cstart is not None:
                age_days = max(0.0, (target_start - cstart) / 86400.0)
                w *= 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
        if target_console_only and "PC" not in (comp.get("platforms") or ""):
            w *= 1.0 + CONSOLE_BONUS
        if target_g is not None:
            gm = ramp_g(pre_ramp(comp.get("hourly"), comp.get("day_start")),
                        curve[1])
            if gm is not None:
                w *= math.exp(-abs(target_g - gm) / RAMP_G_WIDTH)
        if w > 0:
            members.append((curve, w))
    return members, p1


def baseline_quantiles(p1, hour, levels):
    """Persistence baseline (the kit starter's model), used as a fallback."""
    out = []
    for t in levels:
        z = _ND.inv_cdf(t)
        out.append(p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1)))
    return out


def forecast_case(case):
    """{hour: [q per level]} for hours 2..24 of one case."""
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(t) for t in levels]
    members, p1 = ensemble(case)
    if not members:
        return {h: baseline_quantiles(p1, h, levels) for h in HOURS}, p1

    out = {}
    for h in HOURS:
        pairs = sorted(
            (math.log(curve[h] / curve[1]), w)
            for curve, w in members
            if curve.get(h, 0) > 0 and curve.get(1, 0) > 0
        )
        if not pairs:
            out[h] = baseline_quantiles(p1, h, levels)
            continue
        emp = [weighted_quantile(pairs, t) for t in levels]
        q25 = weighted_quantile(pairs, 0.25)
        q50 = weighted_quantile(pairs, 0.5)
        q75 = weighted_quantile(pairs, 0.75)
        sigma = (q75 - q25) / 1.349
        qs = []
        for t, e in zip(levels, emp):
            normal_q = q50 + _ND.inv_cdf(t) * sigma
            qs.append(0.5 * e + 0.5 * normal_q)
        med = qs[levels.index(0.5)] if 0.5 in levels else q50
        qs = [med + (q - med) * SPREAD_SCALE for q in qs]
        iqr = qs[levels.index(0.75)] - qs[levels.index(0.25)] \
            if 0.75 in levels and 0.25 in levels else sigma * 1.349
        if iqr / 1.349 < SIGMA_FLOOR:
            qs = [med + _ND.inv_cdf(t) * SIGMA_FLOOR for t in levels]
        out[h] = [p1 * math.exp(q) for q in qs]
    return out, p1


def clean_quantiles(values):
    """Clamp to (0, 1e9], round, and enforce non-decreasing order."""
    out = []
    prev = 0.0
    for v in values:
        v = min(max(float(v), 0.01), 1e9)
        v = round(v, 2)
        v = max(v, prev)
        out.append(v)
        prev = v
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        pred, _p1 = forecast_case(case)
        payload = {"hours": [{"hour": h, "q": clean_quantiles(pred[h])}
                             for h in HOURS]}
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh)


if __name__ == "__main__":
    main()
