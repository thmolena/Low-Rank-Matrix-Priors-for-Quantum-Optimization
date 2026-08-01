#!/usr/bin/env python3
"""Validate the negative-audit manuscript and reproduction package."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
RESULTS = CODE / "results"
EXPECTED_ROOT = {
    ".gitignore",
    "LICENSE",
    "README.md",
    "index.html",
    "main.tex",
    "main.pdf",
    "code",
}
FIGURES = {
    "kernel_cost_control.pdf",
    "signature_generalization.pdf",
    "tie_neighbor_audit.pdf",
}
INCLUDED_FIGURES = {
    "signature_generalization.pdf",
    "tie_neighbor_audit.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, label: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL {label}")
    print(f"PASS {label}")


def pdf_pages(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        raise SystemExit("FAIL pdfinfo unavailable")
    output = subprocess.check_output([executable, str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    if match is None:
        raise SystemExit("FAIL cannot read PDF page count")
    return int(match.group(1))


def main() -> None:
    root_names = {path.name for path in REPOSITORY.iterdir() if path.name != ".git"}
    require(root_names == EXPECTED_ROOT, "exact public root allowlist")

    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    require(
        manuscript.startswith(r"\documentclass[10pt,letterpaper,twoside]{article}"),
        "one-column preprint class",
    )
    require(
        r"\usepackage{code/manuscript_assets/researchpreprint}" in manuscript,
        "shared preprint style",
    )
    pages = pdf_pages(REPOSITORY / "main.pdf")
    require(3 <= pages <= 12, f"PDF page count 3--12 (found {pages})")
    require(manuscript.count(r"\begin{table") == 1, "exactly one evidence table")
    require(manuscript.count(r"\begin{figure") == 2, "exactly two figures")

    included = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript))
    expected_paths = {f"code/results/figures/{name}" for name in INCLUDED_FIGURES}
    require(included == expected_paths, "exact evidence figure allowlist")
    require(
        {path.name for path in (RESULTS / "figures").glob("*.pdf")} == FIGURES,
        "exact generated figure files",
    )
    figure_source = (CODE / "src/hqml_lie_residual/figures.py").read_text()
    require(".bar(" not in figure_source, "line-only figure implementation")

    bibliography = (CODE / "manuscript_assets/references.bib").read_text()
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript):
        cited.update(key.strip() for key in group.split(","))
    available = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    require(cited <= available, f"all {len(cited)} cited works resolve")
    require(len(cited) >= 2, "primary-literature references retained")

    normalized_manuscript = " ".join(manuscript.split())
    for phrase in (
        "complete negative result is part of the claim",
        "no evidence that tie completion itself improves extrapolation",
        "coarse-grid parameter label",
    ):
        require(phrase in normalized_manuscript, f"scope statement: {phrase}")

    result = json.loads((RESULTS / "results.json").read_text())
    require(result["schema_version"] == 2, "results schema version 2")
    require(result["dataset_sha256"] == "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665", "G1 digest pinned")
    require(result["g81_dataset_sha256"] == "74e69d2f5228774cedbdb86da14debf08023556f1d7693b7346ca13df7594d5a", "G81 digest pinned")
    require(result["unique_angle_grid_labels"] == 5, "five-label concentration retained")

    folds = result["cross_validation"]
    mean_knn = sum(row["knn_loss"] for row in folds) / len(folds)
    mean_blend = sum(row["learned_loss"] for row in folds) / len(folds)
    require(mean_knn < mean_blend, "negative baseline conclusion retained")
    require(any(row["learned_loss"] > row["ridge_loss"] for row in folds), "adverse ridge fold retained")
    require(any(row["rcol_selected_alpha"] == 0.0 for row in folds), "prior rejection retained")
    require(all(row["normalization_bound_observed_coverage"] for row in folds), "normalization bounds cover")
    require(all(row["two_rank_bound_observed_coverage"] for row in folds), "conditional two-rank bounds cover")
    require(all(row["qaoa_regret_bound_max_ratio"] <= 1.0 for row in folds), "QAOA regret bounds cover")
    require(
        any(row["sampled_relative_error"] < row["full_operator_frobenius_relative_error"] for row in result["rank_frontier"]),
        "sampled diagnostic noncertificate counterexample retained",
    )
    require(
        all(row["full_relative_error"] < 5e-4 for row in result["runtime_audit"]),
        "runtime-sweep output accuracy",
    )
    require(
        all(row["dense_apply_seconds"] < row["hierarchy_apply_seconds"] for row in result["runtime_audit"]),
        "dense runtime win retained at every audited size",
    )
    require("g81_timing_speedup" not in result, "streamed speedup headline removed")

    require(
        (RESULTS / "results.json").read_bytes()
        == (RESULTS / "locked_results.json").read_bytes(),
        "locked results byte-identical",
    )
    require(
        (RESULTS / "results.json").read_bytes()
        == (CODE / "src/hqml_lie_residual/results/results.json").read_bytes(),
        "wheel results byte-identical",
    )
    require(not (RESULTS / "tables").exists(), "recursive generated-table debris absent")
    require(not list(RESULTS.glob("*_rows.tex")), "generated table-row files absent")

    debris = []
    for path in REPOSITORY.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.name in {"__pycache__", ".pytest_cache", "build", "dist"}:
            debris.append(str(path.relative_to(REPOSITORY)))
        elif path.is_file() and path.suffix in {".pyc", ".aux", ".blg", ".log", ".out", ".xdv"}:
            debris.append(str(path.relative_to(REPOSITORY)))
        elif path.name.endswith(".egg-info"):
            debris.append(str(path.relative_to(REPOSITORY)))
    require(not debris, f"no release debris ({debris})")

    print(
        "release contract: PASS "
        f"pages={pages} citations={len(cited)} figures=2 tables=1 "
        f"results_sha256={sha256(RESULTS / 'results.json')}"
    )


if __name__ == "__main__":
    main()
