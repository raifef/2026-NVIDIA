#!/usr/bin/env python3
"""
Recover Phase-B results into PRD-useful plots.

This script separates:
  (1) quantum sampling runtime (sample_elapsed_s_b)
  (2) classical MTS time-to-hit-target (mts_t_hit_s)
  (3) total wall-clock TTS (tts_hit_s)

It also computes baseline-anchored fast outliers the PRD way:
  tau04(N) = Q0.04 of the DCQO distribution for a chosen metric
  fast_outlier = metric <= tau04(N)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Stats helpers
# -----------------------
def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95%)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def ecdf_steps(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def jitter(x: np.ndarray, scale: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(0)
    return x + rng.normal(0.0, scale, size=x.shape)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_proportion_ztest_one_sided(k1: int, n1: int, k2: int, n2: int, alternative: str = "greater") -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom <= 0:
        return float("nan")
    z = (p1 - p2) / math.sqrt(denom)
    if alternative == "greater":
        return _norm_sf(z)
    if alternative == "less":
        return _norm_sf(-z)
    raise ValueError("alternative must be 'greater' or 'less'")


def fisher_exact_one_sided(k1: int, n1: int, k2: int, n2: int, alternative: str = "greater") -> float:
    try:
        from scipy.stats import fisher_exact  # type: ignore
    except Exception:
        return float("nan")
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative=alternative)[1])


def rank_tau_threshold(x: np.ndarray, q: float = 0.04) -> tuple[float, int, bool]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"), 0, False)
    xs = np.sort(x)
    k = int(math.ceil(q * n))
    k = max(1, min(k, n))
    tau = float(xs[k - 1])
    tie = (k < n) and (xs[k] == xs[k - 1])
    return (tau, k, tie)


def fast_count_with_rank_tau(x: np.ndarray, tau: float, tie_at_tau: bool) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not np.isfinite(tau):
        return 0
    if tie_at_tau:
        return int(np.sum(x < tau))
    return int(np.sum(x <= tau))


def _metric_label(metric: str) -> str:
    labels = {
        "mts_t_hit_s": "MTS t_hit (s)",
        "tts_hit_s": "TTS_hit (s)",
        "sample_elapsed_s_b": "sample_elapsed_s_b (s)",
        "mts_evals_to_hit": "MTS evals to hit",
        "mts_evals_total": "MTS evals total",
    }
    return labels.get(metric, metric)


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phaseb", required=True, help="Phase B full CSV (e.g. phase_b_fullnew.csv)")
    ap.add_argument("--outdir", default="prd_plots", help="Output directory for plots")
    ap.add_argument(
        "--metric",
        default="mts_t_hit_s",
        help="Metric column for outliers (e.g. mts_t_hit_s, tts_hit_s, mts_evals_to_hit)",
    )
    ap.add_argument("--tail_p", type=float, default=0.10, help="Tail fraction to visualize in ECDF plots")
    ap.add_argument("--q_outlier", type=float, default=0.04, help="Outlier quantile threshold (e.g. 0.04)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.phaseb)
    df["method"] = df["method"].astype(str)

    for c in ["N", "steps", "shots", "success", "seed_bestE", "target_e"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "sample_elapsed_s_b" not in df.columns and "sample_elapsed_s" in df.columns:
        df["sample_elapsed_s_b"] = df["sample_elapsed_s"]

    required = ["method", "N", "success", "seed_bestE", "target_e", args.metric]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in phaseB CSV: {missing}")

    if args.metric in df.columns:
        df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    df["seed_hit"] = (df["seed_bestE"] <= df["target_e"]).astype(int)
    df["mts_needed"] = (df["seed_hit"] == 0).astype(int)

    df_succ = df[df["success"] == 1].copy()

    Ns = sorted(df_succ["N"].dropna().unique().astype(int).tolist())
    methods = ["DAQO", "DCQO"]

    # Console summary
    print("\n=== Summary per N (success-only) ===")
    for N in Ns:
        print(f"\nN={N}")
        for m in methods:
            sub = df_succ[(df_succ["N"] == N) & (df_succ["method"] == m)]
            if sub.empty:
                continue
            med_total = float(np.nanmedian(sub.get("tts_hit_s", np.nan)))
            med_samp = float(np.nanmedian(sub.get("sample_elapsed_s_b", np.nan)))
            med_mts = float(np.nanmedian(sub.get("mts_t_hit_s", np.nan)))
            seed_rate = float(np.nanmean(sub["seed_hit"]))
            print(f"  {m:4s}  median tts={med_total:8.3f}s  sample={med_samp:8.3f}s  mts={med_mts:8.4f}s  seed_hit={seed_rate:5.2%}")

    # Baseline-anchored tau04(N)
    tau_by_N: Dict[int, float] = {}
    for N in Ns:
        dcqo = df_succ[(df_succ["N"] == N) & (df_succ["method"] == "DCQO")]
        x = dcqo[args.metric].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        tau_by_N[N] = float(np.quantile(x, args.q_outlier)) if x.size else float("nan")

    df_succ["tau_dcqo_metric"] = df_succ["N"].apply(lambda n: tau_by_N.get(int(n), float("nan")))
    df_succ["fast_outlier_metric"] = (df_succ[args.metric] <= df_succ["tau_dcqo_metric"]).astype(int)

    # Plot 1: runtime decomposition (if columns exist)
    if "sample_elapsed_s_b" in df_succ.columns and "mts_t_hit_s" in df_succ.columns:
        fig, ax = plt.subplots()
        x = np.arange(len(Ns), dtype=float)
        w = 0.35
        for i, m in enumerate(methods):
            xs = x + (i - 0.5) * w
            samp_meds = []
            mts_meds = []
            for N in Ns:
                sub = df_succ[(df_succ["N"] == N) & (df_succ["method"] == m)]
                samp_meds.append(float(np.nanmedian(sub["sample_elapsed_s_b"])))
                mts_meds.append(float(np.nanmedian(sub["mts_t_hit_s"])))
            samp_meds = np.array(samp_meds)
            mts_meds = np.array(mts_meds)
            ax.bar(xs, samp_meds, width=w * 0.95, label=f"{m}: sampling")
            ax.bar(xs, mts_meds, width=w * 0.95, bottom=samp_meds, label=f"{m}: MTS-to-hit", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("N")
        ax.set_ylabel("Median time (s)")
        ax.set_title("Median wall-time decomposition: sampling + MTS-to-hit-target")
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "runtime_decomposition_vs_N.png"), dpi=200)
        plt.close(fig)

    # Plot 2: sampling runtime vs N
    if "sample_elapsed_s_b" in df_succ.columns:
        fig, ax = plt.subplots()
        for m in methods:
            sub = df_succ[df_succ["method"] == m]
            ax.scatter(jitter(sub["N"].to_numpy(dtype=float), 0.02), sub["sample_elapsed_s_b"].to_numpy(dtype=float), alpha=0.35, label=f"{m} (samples)")
            med = sub.groupby("N")["sample_elapsed_s_b"].median()
            ax.plot(med.index.astype(int).to_numpy(), med.to_numpy(), marker="o", label=f"{m} median")
        ax.set_xlabel("N")
        ax.set_ylabel("sample_elapsed_s_b (s)")
        ax.set_title("Quantum sampling runtime")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sampling_runtime_vs_N.png"), dpi=200)
        plt.close(fig)

    # Plot 3: MTS time-to-hit vs N
    if "mts_t_hit_s" in df_succ.columns:
        fig, ax = plt.subplots()
        for m in methods:
            sub = df_succ[df_succ["method"] == m].copy()
            solved_by_seed = sub[sub["seed_hit"] == 1]
            needed_mts = sub[sub["seed_hit"] == 0]
            ax.scatter(jitter(needed_mts["N"].to_numpy(dtype=float), 0.02), needed_mts["mts_t_hit_s"].to_numpy(dtype=float), alpha=0.35, label=f"{m} (needed MTS)")
            ax.scatter(jitter(solved_by_seed["N"].to_numpy(dtype=float), 0.02), solved_by_seed["mts_t_hit_s"].to_numpy(dtype=float), alpha=0.35, marker="x", label=f"{m} (seed-hit)")
            med = sub.groupby("N")["mts_t_hit_s"].median()
            ax.plot(med.index.astype(int).to_numpy(), med.to_numpy(), marker="o", label=f"{m} median")
        ax.set_xlabel("N")
        ax.set_ylabel("mts_t_hit_s (s)")
        ax.set_title("Classical MTS time-to-hit-target")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "mts_t_hit_vs_N.png"), dpi=200)
        plt.close(fig)

    # Plot 4: Seed-hit rate vs N
    fig, ax = plt.subplots()
    for m in methods:
        xs, ps, lo, hi = [], [], [], []
        for N in Ns:
            sub = df_succ[(df_succ["N"] == N) & (df_succ["method"] == m)]
            n = int(sub.shape[0]); k = int(sub["seed_hit"].sum())
            p = k / n if n else float("nan")
            ci_lo, ci_hi = wilson_ci(k, n)
            xs.append(N); ps.append(p); lo.append(ci_lo); hi.append(ci_hi)
        xs = np.array(xs, dtype=float)
        ps = np.array(ps, dtype=float)
        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], marker="o", capsize=3, linestyle="-", label=m)
    ax.set_xlabel("N")
    ax.set_ylabel("P(seed_bestE <= target_e)")
    ax.set_title("Seed-hit probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "seed_hit_rate_vs_N.png"), dpi=200)
    plt.close(fig)

    # Plot 5: Fast-outlier rate vs N for chosen metric
    fig, ax = plt.subplots()
    for m in methods:
        xs, ps, lo, hi = [], [], [], []
        for N in Ns:
            sub = df_succ[(df_succ["N"] == N) & (df_succ["method"] == m)]
            n = int(sub.shape[0])
            k = int(sub["fast_outlier_metric"].sum())
            p = k / n if n else float("nan")
            ci_lo, ci_hi = wilson_ci(k, n)
            xs.append(N); ps.append(p); lo.append(ci_lo); hi.append(ci_hi)
        xs = np.array(xs, dtype=float)
        ps = np.array(ps, dtype=float)
        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], marker="o", capsize=3, linestyle="-", label=f"{m}")
    ax.set_xlabel("N")
    ax.set_ylabel(f"P({args.metric} <= tau04_DCQO(N))")
    ax.set_title(f"Fast-outlier rate (tau04 from DCQO; metric={args.metric})")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fast_outlier_rate_vs_N_{args.metric}.png"), dpi=200)
    plt.close(fig)

    # Plot 6: Tail ECDF per N for chosen metric
    for N in Ns:
        fig, ax = plt.subplots()
        dcqo = df_succ[(df_succ["N"] == N) & (df_succ["method"] == "DCQO")]
        xdc = dcqo[args.metric].to_numpy(dtype=float)
        xdc = xdc[np.isfinite(xdc)]
        if xdc.size == 0:
            continue
        tail_cut = float(np.quantile(xdc, args.tail_p))
        tau = tau_by_N[N]
        for m in methods:
            sub = df_succ[(df_succ["N"] == N) & (df_succ["method"] == m)]
            x = sub[args.metric].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            x = x[x <= tail_cut]
            xs, ys = ecdf_steps(x)
            if xs.size:
                ax.step(xs, ys, where="post", label=m)
        ax.axvline(tau, linestyle="--", label="DCQO tau04")
        ax.set_xlabel(_metric_label(args.metric))
        ax.set_ylabel("ECDF (tail)")
        ax.set_title(f"Tail ECDF (<= {int(args.tail_p*100)}%)  N={N}  tau04={tau:.4g}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"tail_ecdf_{args.metric}_N{N}.png"), dpi=200)
        plt.close(fig)

    # Add-on analyses: seed-hit tests and conditional tails
    seed_rows: List[Dict[str, float]] = []
    cond_rows: List[Dict[str, float]] = []
    for N in Ns:
        subN = df_succ[df_succ["N"] == N]
        dc = subN[subN["method"] == "DCQO"]
        da = subN[subN["method"] == "DAQO"]
        k_dc, n_dc = int(dc["seed_hit"].sum()), int(dc.shape[0])
        k_da, n_da = int(da["seed_hit"].sum()), int(da.shape[0])
        pz = two_proportion_ztest_one_sided(k_dc, n_dc, k_da, n_da, alternative="greater")
        pf = fisher_exact_one_sided(k_dc, n_dc, k_da, n_da, alternative="greater")
        seed_rows.append({
            "metric": "seed_hit_rate",
            "N": int(N),
            "k_dcqo": k_dc, "n_dcqo": n_dc,
            "k_daqo": k_da, "n_daqo": n_da,
            "pval_z_one_sided_dcqo_gt_daqo": pz,
            "pval_fisher_one_sided_dcqo_gt_daqo": pf,
        })

        if "mts_t_hit_s" in df_succ.columns:
            condN = df_succ[(df_succ["N"] == N) & (df_succ["mts_needed"] == 1)]
            dc = condN[condN["method"] == "DCQO"]
            da = condN[condN["method"] == "DAQO"]
            x_dc = dc["mts_t_hit_s"].to_numpy(dtype=float)
            x_da = da["mts_t_hit_s"].to_numpy(dtype=float)
            x_dc = x_dc[np.isfinite(x_dc)]
            x_da = x_da[np.isfinite(x_da)]
            tau, k_rank, tie_at_tau = rank_tau_threshold(x_dc, q=args.q_outlier)
            k_fast_dc = fast_count_with_rank_tau(x_dc, tau, tie_at_tau)
            k_fast_da = fast_count_with_rank_tau(x_da, tau, tie_at_tau)
            n_dc2 = int(x_dc.size)
            n_da2 = int(x_da.size)
            pz2 = two_proportion_ztest_one_sided(k_fast_dc, n_dc2, k_fast_da, n_da2, alternative="greater")
            pf2 = fisher_exact_one_sided(k_fast_dc, n_dc2, k_fast_da, n_da2, alternative="greater")
            cond_rows.append({
                "metric": "fast_outlier_rate_conditional_rank_tau04",
                "N": int(N),
                "tau_rank": float(tau),
                "k_fast_dcqo": int(k_fast_dc), "n_dcqo": int(n_dc2),
                "k_fast_daqo": int(k_fast_da), "n_daqo": int(n_da2),
                "pval_z_one_sided_dcqo_gt_daqo": pz2,
                "pval_fisher_one_sided_dcqo_gt_daqo": pf2,
            })

    if seed_rows or cond_rows:
        tests_df = pd.DataFrame(seed_rows + cond_rows)
        tests_df.to_csv(os.path.join(args.outdir, "fast_basin_hypothesis_tests.csv"), index=False)

    # Bonus: total TTS scatter with outliers highlighted
    if "tts_hit_s" in df_succ.columns:
        fig, ax = plt.subplots()
        for m in methods:
            sub = df_succ[df_succ["method"] == m]
            ax.scatter(jitter(sub["N"].to_numpy(dtype=float), 0.02), sub["tts_hit_s"].to_numpy(dtype=float), alpha=0.25, label=f"{m} (all)")
            out = sub[sub["fast_outlier_metric"] == 1]
            ax.scatter(jitter(out["N"].to_numpy(dtype=float), 0.02), out["tts_hit_s"].to_numpy(dtype=float), alpha=0.9, edgecolors="black", linewidths=0.7, label=f"{m} (fast by {args.metric})")
        ax.set_xlabel("N")
        ax.set_ylabel("tts_hit_s (s)")
        ax.set_title(f"Total TTS vs N (outliers defined on {args.metric}, tau04 from DCQO)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"tts_vs_N_outliers_by_{args.metric}.png"), dpi=200)
        plt.close(fig)

    print(f"\nWrote plots to: {args.outdir}")


if __name__ == "__main__":
    main()
