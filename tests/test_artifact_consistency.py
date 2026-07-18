"""Round-9 artifact-integrity guards.

These lock the couplings that silently rot: the figures the paper ships must be
byte-identical to the ones in results/, the test count the paper quotes must be
the real one, and a handful of load-bearing numbers must actually appear in the
manuscript. A drift in any of these fails CI rather than reaching a reviewer.
"""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_FIG = ROOT / "paper" / "figures"
RESULT_FIG = ROOT / "results" / "figures"
TESTS = ROOT / "tests"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_paper_figures_match_results_figures():
    """Every figure committed under paper/figures must be byte-identical to the
    generated one in results/figures (no stale copy shipped in the PDF)."""
    mismatches = []
    for pf in sorted(PAPER_FIG.glob("*.png")):
        rf = RESULT_FIG / pf.name
        if rf.exists() and _sha(pf) != _sha(rf):
            mismatches.append(pf.name)
    assert not mismatches, (
        f"paper/figures out of sync with results/figures: {mismatches}. "
        "Re-copy results/figures/<name>.png -> paper/figures/ after regenerating.")


def _claimed_test_count(text: str) -> int | None:
    m = re.search(r"pytest[^)]*?(\d+)\s+tests", text)
    return int(m.group(1)) if m else None


def test_claimed_test_count_matches_actual():
    """The 'pytest -q, N tests' the supplement quotes must equal the real
    pytest-COLLECTED count (parametrized items included -- what a user sees),
    obtained from a --collect-only subprocess (no recursion; it does not run)."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "--collect-only", "-q"],
        capture_output=True, text=True, cwd=str(ROOT))
    m = re.search(r"(\d+)\s+tests? collected", out.stdout)
    assert m, f"could not parse collected count from:\n{out.stdout[-500:]}"
    actual = int(m.group(1))
    supp = (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    claimed = _claimed_test_count(supp)
    assert claimed is not None, "no 'pytest -q, N tests' phrase found in supplement"
    assert claimed == actual, (
        f"supplement claims {claimed} tests but pytest collects {actual}")


def test_key_counts_appear_in_paper():
    """Load-bearing integer counts must literally appear in the manuscript, so a
    JSON change that isn't reflected in the text fails here."""
    import json
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    supp = (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    both = main + "\n" + supp
    cyc = json.loads((ROOT / "results" / "real_cyclone.json").read_text())
    wc = cyc["wind_label_counts"]
    # raw fixes and storm count must be stated somewhere in the paper
    for token in (str(wc["raw_fixes"]), str(wc["raw_storms"])):
        assert token in both, f"count {token} (IBTrACS) not found in the paper text"
