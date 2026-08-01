"""Evidence-only line figures for the negative audit manuscript."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PDF_METADATA = {
    "Creator": "hqml_lie_residual",
    "CreationDate": datetime(2026, 7, 31, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 31, tzinfo=timezone.utc),
}


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(figure)


def write_figures(result: dict, output: Path) -> None:
    """Write exactly four figures derived from the recorded result dictionary."""

    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.pdf"):
        stale.unlink()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
        }
    )

    folds = result["cross_validation"]
    x = np.arange(1, len(folds) + 1)
    methods = [
        ("constant", "baseline", "#777777", "o"),
        ("ridge", "ridge", "#4c72b0", "s"),
        ("kNN", "knn", "#55a868", "^"),
        ("tie-complete neighbor", "tie_knn", "#2a9d8f", "P"),
        ("kernel ridge", "kernel_ridge", "#8172b3", "D"),
        ("rank-stability blend", "learned", "#c44e52", "v"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))
    for label, key, color, marker in methods:
        axes[0].plot(
            x,
            [row[f"{key}_loss"] for row in folds],
            marker=marker,
            color=color,
            linewidth=1.25,
            label=label,
        )
        axes[1].plot(
            x,
            [row[f"{key}_task_error"] for row in folds],
            marker=marker,
            color=color,
            linewidth=1.25,
            label=label,
        )
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[0].set(xlabel="held-out fold", ylabel="angle MSE", xticks=x)
    axes[1].set(xlabel="held-out fold", ylabel="normalized QAOA regret", xticks=x)
    axes[0].legend(frameon=False, ncol=2)
    _save(figure, output / "fig_baseline_audit.pdf")

    figure, axes = plt.subplots(1, 4, figsize=(12.2, 2.75))
    focus = [
        ("ordinary kNN", "knn", "#777777", "o"),
        ("tie-complete", "tie_knn", "#2a9d8f", "P"),
        ("rank-stability blend", "learned", "#c44e52", "v"),
    ]
    for label, key, color, marker in focus:
        axes[0].plot(x, [row[f"{key}_loss"] for row in folds], marker=marker, color=color, label=label)
        axes[1].plot(x, [row[f"{key}_task_error"] for row in folds], marker=marker, color=color, label=label)
    axes[0].set(xlabel="held-out fold", ylabel="angle MSE", xticks=x, yscale="log")
    axes[1].set(xlabel="held-out fold", ylabel="normalized QAOA regret", xticks=x, yscale="symlog")
    axes[0].grid(alpha=0.23, which="both")
    axes[1].grid(alpha=0.23, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].plot(x, [row["tie_knn_neighbors"] for row in folds], "o-", color="#4c72b0", label="selected k")
    axes[2].plot(x, [row["tie_knn_effective_neighbors"] for row in folds], "s-", color="#2a9d8f", label="mean effective count")
    axes[2].set(xlabel="held-out fold", ylabel="neighbors", xticks=x)
    axes[2].grid(alpha=0.23)
    axes[2].legend(frameon=False, fontsize=7)
    tie_runtime = result["tie_knn_runtime"]
    axes[3].plot(
        [row["reference_records"] for row in tie_runtime],
        [1e6 * row["median_seconds"] for row in tie_runtime],
        "o-", color="#2a9d8f", label="measured",
    )
    reference = tie_runtime[0]
    axes[3].plot(
        [row["reference_records"] for row in tie_runtime],
        [1e6 * reference["median_seconds"] * row["distance_products"] / reference["distance_products"] for row in tie_runtime],
        "--", color="#777777", label="distance-work scaling",
    )
    axes[3].set(xlabel="training records", ylabel=r"median prediction time ($\mu$s)")
    axes[3].grid(alpha=0.23)
    axes[3].legend(frameon=False, fontsize=7)
    _save(figure, output / "tie_neighbor_audit.pdf")

    signature_folds = result["signature_group_cross_validation"]
    sx = np.arange(1, len(signature_folds) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(9.7, 2.65))
    for label, key, color, marker in (
        ("constant", "constant", "#777777", "o"),
        ("ordinary kNN", "knn", "#55a868", "^"),
        ("tie-complete", "tie_knn", "#2a9d8f", "P"),
    ):
        axes[0].plot(sx, [row[f"{key}_loss"] for row in signature_folds], marker=marker, color=color, label=label)
        axes[1].plot(sx, [row[f"{key}_task_error"] for row in signature_folds], marker=marker, color=color, label=label)
    axes[0].set(xlabel="held-out signature fold", ylabel="angle MSE", xticks=sx, yscale="log")
    axes[1].set(xlabel="held-out signature fold", ylabel="normalized QAOA regret", xticks=sx, yscale="symlog")
    for axis in axes:
        axis.grid(alpha=0.23, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].semilogy(
        sx,
        [max(row["tie_knn_task_error"], 1e-12) for row in signature_folds],
        "o-", color="#2a9d8f", label="observed regret",
    )
    axes[2].semilogy(
        sx,
        [row["tie_knn_qaoa_bound_mean"] for row in signature_folds],
        "s--", color="#c44e52", label="commutator bound",
    )
    axes[2].set(xlabel="held-out signature fold", ylabel="regret or bound", xticks=sx)
    axes[2].legend(frameon=False, fontsize=7)
    _save(figure, output / "signature_generalization.pdf")

    frontier = result["rank_frontier"]
    ranks = np.asarray([row["rank"] for row in frontier])
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))
    axes[0].plot(
        ranks,
        [row["sampled_relative_error"] for row in frontier],
        "-o",
        linewidth=1.3,
        label="sampled action",
    )
    axes[0].plot(
        ranks,
        [row["full_operator_frobenius_relative_error"] for row in frontier],
        "-s",
        linewidth=1.3,
        label="full Frobenius",
    )
    axes[0].plot(
        ranks,
        [row["prediction_relative_error"] for row in frontier],
        "-^",
        linewidth=1.3,
        label="normalized prediction",
    )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="hierarchy rank", ylabel="relative error")
    axes[0].legend(frameon=False)
    axes[1].plot(
        ranks,
        [row["storage_fraction"] for row in frontier],
        "-o",
        linewidth=1.3,
        label="storage/dense",
    )
    axes[1].plot(
        ranks,
        [1.0 / row["counted_work_ratio"] for row in frontier],
        "-s",
        linewidth=1.3,
        label="counted work/dense",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="hierarchy rank", ylabel="resource fraction")
    axes[1].legend(frameon=False)
    _save(figure, output / "fig_rank_audit.pdf")

    actual = np.asarray(
        [row["hierarchy_to_dense_regret_difference"] for row in folds]
    )
    bounds = np.asarray(
        [row["hierarchy_to_dense_regret_bound"] for row in folds]
    )
    qaoa_bounds = np.asarray([row["qaoa_regret_bound_mean"] for row in folds])
    qaoa_regret = np.asarray([row["learned_task_error"] for row in folds])
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))
    axes[0].plot(x, actual, "-o", linewidth=1.3, label="observed difference")
    axes[0].plot(x, bounds, "-s", linewidth=1.3, label="normalization chain bound")
    axes[0].set_yscale("symlog", linthresh=1e-7)
    axes[0].set(
        xlabel="held-out fold",
        ylabel="hierarchy vs dense regret difference",
        xticks=x,
    )
    axes[0].legend(frameon=False)
    axes[1].plot(x, qaoa_regret, "-o", linewidth=1.3, label="observed regret")
    axes[1].plot(x, qaoa_bounds, "-s", linewidth=1.3, label="commutator bound")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="held-out fold", ylabel="normalized regret", xticks=x)
    axes[1].legend(frameon=False)
    _save(figure, output / "fig_bound_audit.pdf")

    runtime = result["runtime_audit"]
    sizes = np.asarray([row["records"] for row in runtime])
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))
    axes[0].plot(
        sizes,
        [1e3 * row["hierarchy_apply_seconds"] for row in runtime],
        "-o",
        linewidth=1.3,
        label="hierarchy apply",
    )
    axes[0].plot(
        sizes,
        [1e3 * row["dense_apply_seconds"] for row in runtime],
        "-s",
        linewidth=1.3,
        label="optimized dense apply",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="G81-derived records", ylabel="median apply time (ms)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        sizes,
        [row["full_relative_error"] for row in runtime],
        "-o",
        linewidth=1.3,
        label="full output error",
    )
    axes[1].plot(
        sizes,
        [row["hmat_storage_fraction"] for row in runtime],
        "-s",
        linewidth=1.3,
        label="storage/dense",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="G81-derived records", ylabel="error or fraction")
    axes[1].legend(frameon=False)
    _save(figure, output / "fig_runtime_audit.pdf")

    figure, axes = plt.subplots(1, 3, figsize=(9.5, 2.75))
    axes[0].semilogy(ranks, [row["full_operator_frobenius_relative_error"] for row in frontier], "o-", label="operator error")
    axes[0].semilogy(ranks, [row["prediction_relative_error"] for row in frontier], "s-", label="prediction error")
    axes[0].set(xlabel="hierarchy rank", ylabel="relative error")
    axes[0].grid(alpha=0.23, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].loglog(sizes, [1e3 * row["dense_apply_seconds"] for row in runtime], "o-", label="dense BLAS")
    axes[1].loglog(sizes, [1e3 * row["hierarchy_apply_seconds"] for row in runtime], "s-", label="hierarchy")
    axes[1].set(xlabel="G81-derived records", ylabel="median apply time (ms)")
    axes[1].grid(alpha=0.23, which="both")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].loglog(sizes, [row["hmat_storage_fraction"] for row in runtime], "o-", label="storage/dense")
    axes[2].loglog(sizes, [row["full_relative_error"] for row in runtime], "s-", label="output error")
    axes[2].set(xlabel="G81-derived records", ylabel="fraction or error")
    axes[2].grid(alpha=0.23, which="both")
    axes[2].legend(frameon=False, fontsize=7)
    _save(figure, output / "kernel_cost_control.pdf")
    for obsolete in (
        "fig_baseline_audit.pdf",
        "fig_bound_audit.pdf",
        "fig_rank_audit.pdf",
        "fig_runtime_audit.pdf",
    ):
        (output / obsolete).unlink(missing_ok=True)
