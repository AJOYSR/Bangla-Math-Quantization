"""Answer extraction and scoring tests. No GPU needed: `python -m pytest tests`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import is_correct, is_gradable, last_boxed, normalize  # noqa: E402


def test_last_boxed_nested():
    assert last_boxed(r"so $\boxed{\frac{1}{2}}$ is it") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert last_boxed("no answer here") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_bangla_digits_match_ascii():
    assert is_correct(r"উত্তর $\boxed{১৬}$টি", "16")


def test_multiple_choice_forms():
    assert normalize(r"\text{(B)}") == "B"
    assert is_correct(r"\boxed{\text{A}}", "A")
    assert not is_correct(r"\boxed{C}", "A")


def test_numeric_equivalence():
    assert is_correct(r"\boxed{0.5}", r"\dfrac{1}{2}")
    assert is_correct(r"\boxed{1,000}", "1000")
    assert is_correct(r"\boxed{-\frac{3}{4}}", "-0.75")
    assert not is_correct(r"\boxed{17}", "16")


def test_dataset_specific_formats():
    assert is_correct(r"\boxed{B}", r"\textbf{(B)}60")
    assert is_correct(r"\boxed{60}", r"\textbf{(B)}60")
    assert is_correct(r"\boxed{50}", r"\$50")
    assert is_correct(r"\boxed{1280}", "1280 বর্গমিটার")
    assert is_correct(r"\boxed{\frac{5}{12}}", "5/12")
    assert is_gradable(r"\frac{\sqrt{2}}{2}")
    assert not is_gradable("না, তারা ফিরে যেতে পারবে না।")


def test_missing_box_is_wrong():
    assert not is_correct("the answer is 16", "16")


def test_choice_letter_matches_option_value():
    problem = "সংখ্যা কত?\n\nA: 243\n\nB: 125\n\nC: 60\n\nD: 120"
    assert is_correct(r"\boxed{243}", "A", problem)
    assert is_correct(r"\boxed{A}", "243", problem)
    assert not is_correct(r"\boxed{125}", "A", problem)
    assert not is_correct(r"\boxed{243}", "A")  # without the problem text, no mapping
