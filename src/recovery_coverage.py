"""Prospective repeated-sample bootstrap coverage gate, before human data.

The engineering floors below do not certify exact 95% coverage. They reject
substantial undercoverage without demanding that every single CI contain truth.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scipy.stats import beta as beta_distribution

from psychometrics import fit_b384, fit_c192, fit_d384, parametric_bootstrap

RULE_ID = "prospective-coverage-v1-20261029"
MODELS = ("B192", "B384", "C192", "D384")
PROFILES = ("low-steep", "central", "high-shallow")
CRITERIA = {
    "minimum_outer_replicates": 60,
    "minimum_outer_per_profile": 20,
    "minimum_bootstrap_per_outer": 200,
    "minimum_bootstrap_success_fraction": 0.80,
    "minimum_observed_coverage": 0.90,
    "minimum_one_sided_95_lower_bound": 0.80,
    "minimum_profile_observed_coverage": 0.80,
}


def _lower_bound(hits: int, count: int) -> float:
    return float(beta_distribution.ppf(.05, hits, count - hits + 1)) if hits else 0.0


def summarise_coverage(rows: Sequence[Mapping[str, Any]], bootstrap_count: int) -> dict:
    """Recompute every gate from complete outer records; failed fits count as misses."""
    counts = {name: sum(r.get("profile") == name for r in rows) for name in PROFILES}
    enough = (len(rows) >= CRITERIA["minimum_outer_replicates"]
              and all(v >= CRITERIA["minimum_outer_per_profile"] for v in counts.values())
              and bootstrap_count >= CRITERIA["minimum_bootstrap_per_outer"])
    reasons = [] if enough else ["repeated coverage simulation/bootstrap count insufficient"]
    summaries = {}
    for model in MODELS:
        names = ("c80", "beta") if model == "D384" else ("c80",)
        summaries[model] = {}
        for parameter in names:
            successful = []
            for row in rows:
                m = row.get("models", {}).get(model, {})
                valid = (m.get("fit_converged") is True
                         and isinstance(m.get("bootstrap_successful"), int)
                         and m["bootstrap_successful"] >= math.ceil(.8 * bootstrap_count))
                successful.append(valid and m.get("covered", {}).get(parameter) is True)
            n = len(rows); hits = sum(successful)
            profile_rates = {name: (sum(ok for row, ok in zip(rows, successful)
                                       if row.get("profile") == name) / count if count else 0.0)
                             for name, count in counts.items()}
            fraction = hits / n if n else 0.0
            low = _lower_bound(hits, n) if n else 0.0
            passed = (enough and fraction >= .90 and low >= .80
                      and all(v >= .80 for v in profile_rates.values()))
            summaries[model][parameter] = {"included": hits, "total": n,
                "observed_fraction": fraction, "one_sided_95_lower_bound": low,
                "profile_fractions": profile_rates, "passed": passed}
            if not passed:
                reasons.append(f"{model}.{parameter}: repeated coverage guard failed")
    return {"rule_id": RULE_ID, "criteria": dict(CRITERIA),
            "outer_replicates": len(rows), "bootstrap_per_outer": bootstrap_count,
            "profile_counts": counts, "models": summaries, "passed": not reasons,
            "reasons": reasons,
            "interpretation": "Engineering undercoverage guard, not proof of exact nominal 95% coverage."}


def validate_coverage_payload(payload: Any) -> bool:
    """Fail closed on missing records, duplicates, wrong rule or forged aggregates."""
    if not isinstance(payload, dict):
        return False
    rows = payload.get("outer_records")
    count = payload.get("bootstrap_per_outer")
    if not isinstance(rows, list) or isinstance(count, bool) or not isinstance(count, int):
        return False
    if not rows or any(not isinstance(r, dict) for r in rows):
        return False
    if [r.get("outer_index") for r in rows] != list(range(len(rows))):
        return False
    if any(r.get("profile") != PROFILES[i % 3] for i, r in enumerate(rows)):
        return False
    for row in rows:
        models = row.get("models")
        if not isinstance(models, dict) or set(models) != set(MODELS):
            return False
        for model, entry in models.items():
            parameters = {"c80", "beta"} if model == "D384" else {"c80"}
            if (not isinstance(entry, dict) or not isinstance(entry.get("fit_converged"), bool)
                    or isinstance(entry.get("bootstrap_successful"), bool)
                    or not isinstance(entry.get("bootstrap_successful"), int)
                    or not 0 <= entry["bootstrap_successful"] <= count
                    or not isinstance(entry.get("covered"), dict)
                    or set(entry["covered"]) != parameters
                    or any(not isinstance(v, bool) for v in entry["covered"].values())):
                return False
    recomputed = summarise_coverage(rows, count)
    return recomputed["passed"] and all(payload.get(k) == v for k, v in recomputed.items())


def run_repeated_coverage(*, simulate_trials: Callable, profiles: Sequence[Mapping],
                          widths: Sequence[float], reference: float, trials: Sequence[int],
                          levels: Sequence[float], lapse: float, relative_scale: Callable,
                          seed: int, outer_replicates: int, bootstrap_replicates: int,
                          progress: Callable | None = None, workers: int = 1) -> dict:
    """Fresh seed stream, equally rotated stress profiles; no accepted-seed search."""
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and outer_replicates:
        # Chunk starts are multiples of three, preserving profile and seed order.
        # Workers change execution speed only, never the random streams or gates.
        arguments=[]
        for start in range(0,outer_replicates,6):
            arguments.append((start,dict(simulate_trials=simulate_trials,profiles=profiles,
                widths=widths,reference=reference,trials=trials,levels=levels,lapse=lapse,
                relative_scale=relative_scale,seed=seed+11*start,
                outer_replicates=min(6,outer_replicates-start),
                bootstrap_replicates=bootstrap_replicates,workers=1)))
        rows=[]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk in executor.map(_coverage_chunk,arguments):
                rows.extend(chunk)
                if progress is not None:progress(len(rows),outer_replicates)
        result=summarise_coverage(rows,bootstrap_replicates)
        result.update(outer_records=rows,seed=seed,
            seed_rule="data=seed+900000+11*i; bootstrap=seed+1900000+11*i")
        return result
    rows = []
    for i in range(outer_replicates):
        profile = profiles[i % 3]
        truth = float(profile["c80_ref"]); w = float(profile["w"])
        ds = seed + 900_000 + 11 * i; bs = seed + 1_900_000 + 11 * i
        common = dict(widths=widths, trials_per_width=trials, w=w, lapse=lapse,
                      delta_e00_levels=levels)
        xc, yc, sc = simulate_trials(**common, seed=ds, c80_at_width=lambda _: truth)
        xd, yd, sd = simulate_trials(**common, seed=ds+1,
            c80_at_width=lambda v: truth * (v / reference) ** -.48)
        xt, yt, st = simulate_trials(**common, seed=ds+2,
            c80_at_width=lambda v: truth * relative_scale(v))
        b = fit_b384(xc, yc, sc, lapse=lapse)
        d = fit_d384(xd, yd, sd, reference_width=reference, lapse=lapse)
        c = fit_c192(xt, yt, st, reference_width=reference,
                     relative_scale=relative_scale, lapse=lapse)
        models = {}
        for label, fit, x, y, s, bootstrap_seed in (
                ("B384", b, xc, yc, sc, bs), ("D384", d, xd, yd, sd, bs+1),
                ("C192", c, xt, yt, st, bs+2)):
            try:
                boot = parametric_bootstrap(fit, x, y, s, replicates=bootstrap_replicates,
                                            seed=bootstrap_seed)
                c_key = "c80" if label == "B384" else "c80_ref"
                lo, hi = boot.intervals.get(c_key, (math.inf, -math.inf))
                covered = {"c80": bool(lo <= truth <= hi)}
                if label == "D384":
                    lo, hi = boot.intervals.get("beta", (math.inf, -math.inf))
                    covered["beta"] = bool(lo <= -.48 <= hi)
                models[label] = {"fit_converged": bool(fit.converged),
                    "bootstrap_successful": boot.successful_replicates, "covered": covered}
            except (ValueError, RuntimeError, FloatingPointError):
                models[label] = {"fit_converged": False, "bootstrap_successful": 0,
                    "covered": {k: False for k in (("c80", "beta") if label=="D384" else ("c80",))}}
        # B192 and C192 at their reference use the identical 192 responses and
        # identical likelihood. The separate representative paired test verifies
        # numerical fits and bootstrap draws before this shared coverage result.
        models["B192"] = {**models["C192"], "covered": dict(models["C192"]["covered"])}
        rows.append({"outer_index": i, "profile": str(profile["name"]),
                     "data_seed": ds, "bootstrap_seed": bs, "models": models})
        if progress is not None and ((i+1) % 5 == 0 or i+1 == outer_replicates):
            progress(i+1, outer_replicates)
    result = summarise_coverage(rows, bootstrap_replicates)
    result.update(outer_records=rows, seed=seed,
                  seed_rule="data=seed+900000+11*i; bootstrap=seed+1900000+11*i")
    return result


def _coverage_chunk(argument):
    start,kwargs=argument
    rows=run_repeated_coverage(**kwargs)["outer_records"]
    for row in rows:row["outer_index"]+=start
    return rows
