from __future__ import annotations

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def test_evidence_allowlist_and_no_recursive_table_debris() -> None:
    figures = REPOSITORY / "code" / "results" / "figures"
    assert {path.name for path in figures.glob("*.pdf")} == {
        "tie_neighbor_audit.pdf",
        "signature_generalization.pdf",
        "kernel_cost_control.pdf",
    }
    assert not (REPOSITORY / "code" / "results" / "tables").exists()
    assert not list((REPOSITORY / "code" / "results").glob("*_rows.tex"))


def test_manuscript_has_small_evidence_set_and_scope_claims() -> None:
    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    assert manuscript.count(r"\begin{table") == 1
    assert manuscript.count(r"\begin{figure") == 2
    for phrase in (
        "training and test signatures",
        "it does not improve extrapolation beyond unseen signatures",
        "zero train--test signature overlap",
        "The complete negative result is part of the claim",
    ):
        assert phrase in manuscript
