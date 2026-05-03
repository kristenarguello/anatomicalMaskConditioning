#!/usr/bin/env python3
"""
Build a summary table (mean ± std) across all models and variants.

Reads per_slice_metrics.csv from results/{variant}/ directories and
produces a printed table + LaTeX-formatted version.

Usage:
  python build_summary_table.py
  python build_summary_table.py --results_root /custom/path/results
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(_HERE, "results"))

# Display order and labels for the table
VARIANT_ORDER = [
    # (directory_name,           display_label)
    ("redcnn_no_mask",            "RED-CNN (baseline)"),
    ("redcnn_mask",               "RED-CNN + mask"),
    ("corediff_no_mask_noctx",    "CoreDiff (no ctx)"),
    ("corediff_mask_noctx",       "CoreDiff + mask (no ctx)"),
    ("corediff_no_mask_ctx",      "CoreDiff (ctx)"),
    ("corediff_mask_ctx",         "CoreDiff + mask (ctx)"),
    ("iddpm_no_mask",             "iDDPM (baseline)"),
    ("iddpm_mask",                "iDDPM + mask"),
    ("segguideddiff_no_mask",     "SegGuidedDiff (baseline)"),
    ("segguideddiff_mask",        "SegGuidedDiff + mask"),
]

METRICS = ["psnr", "ssim", "rmse_hu", "cnr",
           "boundary_psnr", "boundary_ssim", "interior_psnr", "interior_ssim"]

METRIC_LABELS = {
    "psnr":          "PSNR (dB)↑",
    "ssim":          "SSIM↑",
    "rmse_hu":       "RMSE↓",
    "cnr":           "CNR↑",
    "boundary_psnr": "Bnd PSNR↑",
    "boundary_ssim": "Bnd SSIM↑",
    "interior_psnr": "Int PSNR↑",
    "interior_ssim": "Int SSIM↑",
}


def load_variant(results_root: str, variant_dir: str):
    csv_path = os.path.join(results_root, variant_dir, "per_slice_metrics.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def fmt(mean, std, prec=3):
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


def latex_fmt(mean, std, prec=3, bold=False):
    s = f"{mean:.{prec}f} \\pm {std:.{prec}f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def best_per_column(rows_data, metric_idx, metrics):
    """Return the row index of the best value for this metric."""
    metric = metrics[metric_idx]
    vals = [r[metric_idx] for r in rows_data if r is not None]
    if not vals:
        return None
    means = [v[0] for v in vals]
    if metric == "rmse_hu":
        return means.index(min(means))
    return means.index(max(means))


def build_table(results_root: str):
    rows = []
    for variant_dir, label in VARIANT_ORDER:
        df = load_variant(results_root, variant_dir)
        if df is None:
            rows.append((label, None))
            continue
        stats = []
        for metric in METRICS:
            if metric not in df.columns:
                stats.append((float("nan"), float("nan")))
            else:
                vals = df[metric].dropna().values
                stats.append((float(np.mean(vals)), float(np.std(vals))))
        rows.append((label, stats))

    # ---- Plain text table ----
    col_w = 22
    header = f"{'Model':<30}"
    for m in METRICS:
        header += f"  {METRIC_LABELS[m]:>{col_w}}"
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)
    for label, stats in rows:
        if stats is None:
            print(f"  {label:<28}  (no results yet)")
            continue
        line = f"  {label:<28}"
        for mean, std in stats:
            if np.isnan(mean):
                line += f"  {'—':>{col_w}}"
            else:
                line += f"  {fmt(mean, std):>{col_w}}"
        print(line)
    print(sep + "\n")

    # ---- LaTeX table ----
    print("% LaTeX table\n")
    print("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    print("\\toprule")
    latex_header = "Model & " + " & ".join(
        f"\\textbf{{{METRIC_LABELS[m]}}}" for m in METRICS
    ) + " \\\\"
    print(latex_header)
    print("\\midrule")

    # Identify best per column (only from rows that have data)
    data_only = [(i, stats) for i, (_, stats) in enumerate(rows) if stats is not None]
    best_idx_per_metric = []
    for m_idx, metric in enumerate(METRICS):
        if not data_only:
            best_idx_per_metric.append(None)
            continue
        vals = [(i, stats[m_idx][0]) for i, stats in data_only
                if not np.isnan(stats[m_idx][0])]
        if not vals:
            best_idx_per_metric.append(None)
            continue
        if metric == "rmse_hu":
            best_idx_per_metric.append(min(vals, key=lambda x: x[1])[0])
        else:
            best_idx_per_metric.append(max(vals, key=lambda x: x[1])[0])

    for row_i, (label, stats) in enumerate(rows):
        if stats is None:
            cells = ["—"] * len(METRICS)
        else:
            cells = []
            for m_idx, (mean, std) in enumerate(stats):
                if np.isnan(mean):
                    cells.append("—")
                else:
                    is_best = (best_idx_per_metric[m_idx] == row_i)
                    prec = 4 if METRICS[m_idx] == "ssim" else 2
                    cells.append(latex_fmt(mean, std, prec=prec, bold=is_best))

        # Insert \midrule between model groups
        if row_i in (2, 6, 8):
            print("\\midrule")

        print(f"  {label} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root", default=RESULTS_ROOT,
        help="Root directory containing per-model result subdirs"
    )
    args = p.parse_args()
    build_table(args.results_root)
