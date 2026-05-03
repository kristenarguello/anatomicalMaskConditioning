"""
Paired sign-flipping permutation test: baseline vs masked variant.

The independent unit is the PATIENT (n=12). Slices within a patient are
averaged first, then a per-patient delta is computed. All 2^12 = 4096 sign
assignments are enumerated for an exact two-sided test.

Usage:
  python run_permutation_test.py \\
      --baseline results/corediff_no_mask_noctx/per_slice_metrics.csv \\
      --masked   results/corediff_mask_noctx/per_slice_metrics.csv \\
      --model    corediff

  python run_permutation_test.py \\
      --baseline results/redcnn_no_mask/per_slice_metrics.csv \\
      --masked   results/redcnn_mask/per_slice_metrics.csv \\
      --model    redcnn \\
      --out_csv  results/permtest_redcnn.csv
"""

import argparse
import csv
import os
import re

import numpy as np
import pandas as pd
from itertools import product as iproduct


METRICS = [
    "psnr", "ssim", "rmse_hu", "cnr",
    "boundary_psnr", "boundary_ssim",
    "interior_psnr", "interior_ssim",
]

# For these metrics, lower is better → flip delta so Δ > 0 always means improvement.
LOWER_IS_BETTER = {"rmse_hu"}


def patient_id(slice_name: str) -> str:
    """Extract patient ID from slice filename, e.g. 'L014_QD_1_1_CT_0001.png' → 'L014'."""
    return slice_name.split("_")[0]


def load_patient_means(path: str) -> pd.DataFrame:
    """Load CSV, group by patient, return one row per patient with metric means."""
    df = pd.read_csv(path)
    df["patient"] = df["slice_name"].map(patient_id)
    return df.groupby("patient")[METRICS].mean()


def sign_flip_test(deltas: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    """
    Exact sign-flipping permutation test.

    Enumerates all 2^n sign assignments (feasible for n ≤ ~20).
    Returns (observed_mean, p_value).
    """
    n = len(deltas)
    observed = float(np.mean(deltas))

    perm_stats = np.empty(2 ** n)
    for i, signs in enumerate(iproduct((-1, 1), repeat=n)):
        perm_stats[i] = np.mean(np.array(signs) * deltas)

    if alternative == "two-sided":
        p = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    elif alternative == "greater":
        p = float(np.mean(perm_stats >= observed))
    elif alternative == "less":
        p = float(np.mean(perm_stats <= observed))
    else:
        raise ValueError(alternative)

    return observed, p


def run_tests(baseline_path: str, masked_path: str, model: str) -> list[dict]:
    b_patients = load_patient_means(baseline_path)
    m_patients = load_patient_means(masked_path)

    common_patients = b_patients.index.intersection(m_patients.index)
    n = len(common_patients)

    if n == 0:
        print("[ERROR] No common patients found.")
        return []

    b = b_patients.loc[common_patients]
    m = m_patients.loc[common_patients]

    header = (
        f"\n=== Paired sign-flip permutation test: {model} "
        f"(n={n} patients, 2^{n}={2**n} permutations) ===\n"
        f"  {'Metric':<18}  {'Δ mean':>9}  {'p-value':>9}  sig   "
        f"{'B mean':>10}  {'M mean':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for metric in METRICS:
        if metric not in b.columns:
            continue

        bv = b[metric].values
        mv = m[metric].values

        valid = ~(np.isnan(bv) | np.isnan(mv))
        if valid.sum() < 2:
            continue
        bv, mv = bv[valid], mv[valid]

        deltas = mv - bv
        if metric in LOWER_IS_BETTER:
            deltas = -deltas  # flip so Δ > 0 ↔ improvement

        effect, p = sign_flip_test(deltas, alternative="two-sided")

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        direction = "(↓ better, Δ flipped)" if metric in LOWER_IS_BETTER else ""

        print(
            f"  {metric:<18}  {effect:>+9.4f}  {p:>9.4f}  {sig:<4}  "
            f"{np.mean(bv):>10.4f}  {np.mean(mv):>10.4f}  {direction}"
        )

        rows.append({
            "model":          model,
            "metric":         metric,
            "n_patients":     int(valid.sum()),
            "baseline_mean":  round(float(np.mean(bv)), 6),
            "masked_mean":    round(float(np.mean(mv)), 6),
            "delta_mean":     round(float(effect), 6),
            "p_value":        round(float(p), 6),
            "significant":    sig,
        })

    print("\n  * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    print("  Δ = mean(mask) − mean(baseline) per patient, then averaged")
    print("  (for rmse_hu the sign is flipped so Δ > 0 still means improvement)\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--masked",   required=True)
    ap.add_argument("--model",    required=True)
    ap.add_argument("--out_csv",  default="")
    args = ap.parse_args()

    rows = run_tests(args.baseline, args.masked, args.model)

    if args.out_csv and rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
