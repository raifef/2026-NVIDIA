
"""
Pytest sanity / numerical QA checks for phase_ab_outliers.py outputs.

How to run
----------
# default paths (read from phase_ab_outliers.CONFIG):
pytest -q

# or override input CSVs / module path:
PHASE_AB_PATH=phase_ab_outliers.py \
PHASE_A_CSV=out_stats/phase_a_samples.csv \
PHASE_B_CSV=out_stats/phase_b_outliers.csv \
pytest -q

What this checks
----------------
- Output CSV schema + basic invariants (types, monotone quantiles, bounds)
- LABS energy invariances (global flip, reversal, alternating sign / "stagger")
- CPU batch energy == scalar energy == (optional) GPU batch energy
- Tabu/memetic search never worsens best energy (convergence sanity)
- Physical bounds: 0 <= E <= sum_{k=1}^{N-1} (N-k)^2

Notes
-----
- Some checks are "soft" and will SKIP if required inputs aren't present.
- This is a QA suite: it tries to continue and collect many discrepancies
  before failing, so you get a helpful summary in one go.
"""

from __future__ import annotations

import os
import math
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None  # type: ignore


# ----------------------------
# Helpers
# ----------------------------

def _load_phase_module() -> Any:
    """
    Load phase_ab_outliers.py as a module so we can call labs_energy, etc.
    """
    cand = os.environ.get("PHASE_AB_PATH", "phase_ab_outliers.py")
    p = Path(cand)
    if not p.exists():
        # common repo layout: scripts/phase_ab_outliers.py
        alt = Path("scripts") / cand
        if alt.exists():
            p = alt
    if not p.exists():
        pytest.skip(f"Phase AB module not found at {cand!r}. Set PHASE_AB_PATH.")
    spec = importlib.util.spec_from_file_location("phase_ab_outliers", str(p))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _e_max(n: int) -> int:
    # E = sum_{k=1}^{n-1} c_k^2 with |c_k| <= n-k.
    n = int(n)
    return int(sum((n - k) ** 2 for k in range(1, n)))


def _flip(bits: np.ndarray) -> np.ndarray:
    # global flip: 0<->1
    return (1 - bits).astype(np.int8, copy=False)


def _reverse(bits: np.ndarray) -> np.ndarray:
    return bits[::-1].astype(np.int8, copy=False)


def _stagger(bits: np.ndarray) -> np.ndarray:
    """
    Multiply spins by (-1)^i. In bit language, that flips every odd position:
      s_i = 2 b_i - 1
      s'_i = (-1)^i s_i
    This corresponds to b'_i = b_i xor (i mod 2).
    """
    b = bits.astype(np.int8, copy=False).copy()
    b[1::2] ^= 1
    return b


def _bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=(int(n),), dtype=np.int8)


@dataclass
class Issue:
    where: str
    msg: str


def _assert_close(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    return bool(abs(a - b) <= (atol + rtol * abs(b)))


def _read_csv(path: str):
    if pd is None:
        pytest.skip("pandas not available")
    p = Path(path)
    if not p.exists():
        pytest.skip(f"CSV not found: {path}")
    return pd.read_csv(p)


def _csv_paths_from_config(mod) -> Tuple[Optional[str], Optional[str]]:
    a = os.environ.get("PHASE_A_CSV")
    b = os.environ.get("PHASE_B_CSV")
    if a or b:
        return a, b
    # Fall back to CONFIG in module
    cfg = getattr(mod, "CONFIG", {}) or {}
    return cfg.get("PHASE_A_CSV"), cfg.get("PHASE_B_CSV")


# ----------------------------
# Core numerical correctness checks (module-level)
# ----------------------------

def _check_energy_invariances(mod, rng: np.random.Generator, n: int, trials: int = 200) -> List[Issue]:
    issues: List[Issue] = []
    labs_energy = getattr(mod, "labs_energy")
    for t in range(trials):
        b = _bits(rng, n)
        e0 = int(labs_energy(b.tolist()))
        for name, tf in [("flip", _flip), ("reverse", _reverse), ("stagger", _stagger)]:
            e1 = int(labs_energy(tf(b).tolist()))
            if e1 != e0:
                issues.append(Issue(f"energy_invariance n={n}", f"{name} broke invariance at trial {t}: {e0} vs {e1}"))
                # don't spam: stop early per transform
                break
        if e0 < 0 or e0 > _e_max(n):
            issues.append(Issue(f"energy_bounds n={n}", f"Energy out of bounds at trial {t}: E={e0}, Emax={_e_max(n)}"))
    return issues


def _check_batch_matches_scalar(mod, rng: np.random.Generator, n: int, k: int = 64, trials: int = 50) -> List[Issue]:
    issues: List[Issue] = []
    labs_energy = getattr(mod, "labs_energy")
    labs_energy_batch = getattr(mod, "labs_energy_batch")
    use_gpu_flag = bool(getattr(mod, "_CUPY_AVAILABLE", False))

    for t in range(trials):
        pop = rng.integers(0, 2, size=(k, n), dtype=np.int8).tolist()
        e_scalar = np.array([labs_energy(bits) for bits in pop], dtype=np.int64)

        e_cpu = labs_energy_batch(pop, use_gpu=False).astype(np.int64)
        if not np.array_equal(e_cpu, e_scalar):
            # locate a mismatch to report
            idx = int(np.argmax(e_cpu != e_scalar))
            issues.append(Issue(f"batch_vs_scalar n={n}", f"CPU mismatch trial {t} at idx {idx}: {e_cpu[idx]} vs {e_scalar[idx]}"))
            break

        if use_gpu_flag:
            e_gpu = labs_energy_batch(pop, use_gpu=True).astype(np.int64)
            if not np.array_equal(e_gpu, e_scalar):
                idx = int(np.argmax(e_gpu != e_scalar))
                issues.append(Issue(f"gpu_vs_scalar n={n}", f"GPU mismatch trial {t} at idx {idx}: {e_gpu[idx]} vs {e_scalar[idx]}"))
                break

    return issues


def _check_tabu_convergence(mod, rng: np.random.Generator, n: int, trials: int = 40) -> List[Issue]:
    issues: List[Issue] = []
    labs_energy = getattr(mod, "labs_energy")
    tabu_search = getattr(mod, "tabu_search")
    TabuConfig = getattr(mod, "TabuConfig")

    for t in range(trials):
        b = _bits(rng, n).tolist()
        e0 = int(labs_energy(b))
        best_bits, best_e = tabu_search(b, TabuConfig(max_iters=64, tenure=6, plateau_limit=25, allow_worsen=True), rng=rng)
        best_e = int(best_e)
        if best_e > e0:
            issues.append(Issue(f"tabu_convergence n={n}", f"Trial {t}: tabu worsened energy {e0} -> {best_e}"))
            break
        # also verify reported best_e matches labs_energy(best_bits)
        e_chk = int(labs_energy(best_bits))
        if e_chk != best_e:
            issues.append(Issue(f"tabu_energy_consistency n={n}", f"Trial {t}: best_e={best_e} but labs_energy(best_bits)={e_chk}"))
            break
    return issues


def _check_memetic_convergence(mod, rng: np.random.Generator, n: int, trials: int = 20) -> List[Issue]:
    issues: List[Issue] = []
    labs_energy_batch = getattr(mod, "labs_energy_batch")
    memetic_tabu_search = getattr(mod, "memetic_tabu_search")
    MTSConfig = getattr(mod, "MTSConfig")
    TabuConfig = getattr(mod, "TabuConfig")

    for t in range(trials):
        pop = rng.integers(0, 2, size=(32, n), dtype=np.int8).tolist()
        e0 = labs_energy_batch(pop, use_gpu=False).astype(np.int64)
        e0_min = int(np.min(e0)) if len(e0) else 0
        cfg = MTSConfig(
            population_size=32,
            pcomb=0.7,
            pmutate=1.0 / float(n),
            target_e=0,  # not expecting to hit; just checking monotonic best
            max_outer_iters=200,
            tabu=TabuConfig(max_iters=48, tenure=6, plateau_limit=25, allow_worsen=True),
        )
        _, best_e, stats = memetic_tabu_search(pop, cfg, rng=rng, time_budget_s=0.2)
        best_e = int(best_e)
        if best_e > e0_min:
            issues.append(Issue(f"memetic_convergence n={n}", f"Trial {t}: best worsened {e0_min} -> {best_e}"))
            break
        if "outer_iters" not in stats or "elapsed_s" not in stats or "best_e" not in stats:
            issues.append(Issue(f"memetic_stats n={n}", f"Trial {t}: missing expected stats keys; got {sorted(stats.keys())}"))
            break
    return issues


# ----------------------------
# CSV output sanity checks (Phase A / Phase B)
# ----------------------------

def _check_phase_a_csv(mod, path: str) -> List[Issue]:
    issues: List[Issue] = []
    df = _read_csv(path)

    required = {
        "method","rep","seed","N","steps","dt","T","cost_scale","cd_scale","mix_scale",
        "shots","sample_elapsed_s","shots_per_s","bestE_sample","q10","q50","q90","status","error"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        return [Issue("phase_a_schema", f"Missing columns: {missing}")]

    # Basic row-level invariants for status=ok
    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        issues.append(Issue("phase_a_rows", "No status=ok rows found."))
        return issues

    for idx, row in ok.iterrows():
        where = f"phase_a_row idx={idx} method={row['method']} N={row['N']} steps={row['steps']}"
        n = int(row["N"])
        emax = _e_max(n)

        # time sanity
        t = float(row["sample_elapsed_s"])
        if not (t > 0 and math.isfinite(t)):
            issues.append(Issue(where, f"sample_elapsed_s not positive/finite: {t}"))
        shots = int(row["shots"])
        sps = float(row["shots_per_s"])
        if t > 0 and math.isfinite(sps):
            exp = shots / max(t, 1e-12)
            if not _assert_close(sps, exp, rtol=2e-2, atol=1e-6):
                issues.append(Issue(where, f"shots_per_s inconsistent: got {sps}, expected ~{exp}"))

        # energy bounds + quantile monotonicity
        bestE = float(row["bestE_sample"])
        q10 = float(row["q10"]); q50 = float(row["q50"]); q90 = float(row["q90"])
        for name, val in [("bestE_sample", bestE), ("q10", q10), ("q50", q50), ("q90", q90)]:
            if not (math.isfinite(val) and val >= -1e-9 and val <= emax + 1e-6):
                issues.append(Issue(where, f"{name} out of bounds: {val} (Emax={emax})"))

        if not (bestE <= q10 + 1e-9 <= q50 + 1e-9 <= q90 + 1e-9):
            issues.append(Issue(where, f"Quantiles not monotone or bestE>q10: best={bestE}, q10={q10}, q50={q50}, q90={q90}"))

        # unique sanity: should not exceed shots (if finite)
        if "unique" in df.columns:
            u = row.get("unique", float("nan"))
            try:
                u = float(u)
            except Exception:
                u = float("nan")
            if math.isfinite(u):
                if u < 0 or u > shots + 1e-6:
                    issues.append(Issue(where, f"unique out of range: {u} (shots={shots})"))

        # error field should be empty-ish for ok
        err = str(row.get("error", "") or "")
        if err.strip():
            issues.append(Issue(where, f"status=ok but error field non-empty: {err!r}"))

    return issues


def _check_phase_b_csv(mod, path: str) -> List[Issue]:
    issues: List[Issue] = []
    df = _read_csv(path)

    required = {
        "method","rep","seed","N","steps","cost_scale","cd_scale",
        "shots","sample_elapsed_s_a","sample_elapsed_s_b",
        "seed_bestE","mts_bestE","mts_elapsed_s","mts_outer_iters"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        return [Issue("phase_b_schema", f"Missing columns: {missing}")]

    for idx, row in df.iterrows():
        where = f"phase_b_row idx={idx} method={row['method']} N={row['N']} steps={row['steps']}"
        n = int(row["N"])
        emax = _e_max(n)

        # bitstring length checks are impossible from CSV (bitstrings not stored).
        # We instead check the energy fields are self-consistent.

        seedE = float(row["seed_bestE"])
        mtsE = float(row["mts_bestE"])
        if not (math.isfinite(seedE) and 0 <= seedE <= emax + 1e-6):
            issues.append(Issue(where, f"seed_bestE out of bounds: {seedE} (Emax={emax})"))
        if not (math.isfinite(mtsE) and 0 <= mtsE <= emax + 1e-6):
            issues.append(Issue(where, f"mts_bestE out of bounds: {mtsE} (Emax={emax})"))
        if math.isfinite(seedE) and math.isfinite(mtsE) and mtsE > seedE + 1e-9:
            issues.append(Issue(where, f"MTS worsened best energy: seed_bestE={seedE} -> mts_bestE={mtsE}"))

        mt = float(row["mts_elapsed_s"])
        if not (mt >= 0 and math.isfinite(mt)):
            issues.append(Issue(where, f"mts_elapsed_s not finite/nonnegative: {mt}"))

        outer = row.get("mts_outer_iters")
        try:
            outer_i = int(float(outer))
        except Exception:
            issues.append(Issue(where, f"mts_outer_iters not parseable int: {outer!r}"))
            continue
        if outer_i < 1:
            issues.append(Issue(where, f"mts_outer_iters < 1: {outer_i}"))

    return issues


# ----------------------------
# Pytest entrypoint
# ----------------------------

def test_phase_ab_numerical_sanity():
    """
    Run a battery of checks and fail once with a consolidated discrepancy summary.
    """
    mod = _load_phase_module()
    rng = np.random.default_rng(0)

    issues: List[Issue] = []

    # Core algorithm checks
    for n in [4, 8, 12, 16]:
        issues += _check_energy_invariances(mod, rng, n, trials=200)
        issues += _check_batch_matches_scalar(mod, rng, n, k=64, trials=30)
        issues += _check_tabu_convergence(mod, rng, n, trials=30)
        issues += _check_memetic_convergence(mod, rng, n, trials=15)

    # Output CSV checks (if present)
    phase_a_csv, phase_b_csv = _csv_paths_from_config(mod)
    if phase_a_csv:
        issues += _check_phase_a_csv(mod, phase_a_csv)
    if phase_b_csv:
        issues += _check_phase_b_csv(mod, phase_b_csv)

    if issues:
        # Build a readable summary (grouped by 'where')
        by_where: Dict[str, List[str]] = {}
        for it in issues:
            by_where.setdefault(it.where, []).append(it.msg)

        lines = []
        lines.append("")
        lines.append("========== PHASE-AB QA DISCREPANCY SUMMARY ==========")
        lines.append(f"Total issues: {len(issues)} across {len(by_where)} categories")
        for where in sorted(by_where.keys()):
            msgs = by_where[where]
            lines.append("")
            lines.append(f"[{where}]  ({len(msgs)} issue(s))")
            # show up to 8 messages per category
            for m in msgs[:8]:
                lines.append(f"  - {m}")
            if len(msgs) > 8:
                lines.append(f"  ... ({len(msgs)-8} more)")
        lines.append("=====================================================")
        pytest.fail("\n".join(lines))
