"""Tests to verify the error collector and main error reporting."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.error_collector import collect


def test_collect_appends_when_errors_provided():
    errors = []
    collect(errors, "preprocessing", "Skipped file", "path=/x.npy")
    assert len(errors) == 1
    assert "[preprocessing]" in errors[0]
    assert "Skipped file" in errors[0]
    assert "/x.npy" in errors[0]


def test_collect_no_detail():
    errors = []
    collect(errors, "analysis", "Run failed", None)
    assert len(errors) == 1
    assert errors[0] == "[analysis] Run failed"


def test_collect_does_nothing_when_errors_none():
    collect(None, "preprocessing", "Msg", "detail")
    # no exception, no side effect


if __name__ == "__main__":
    test_collect_appends_when_errors_provided()
    test_collect_no_detail()
    test_collect_does_nothing_when_errors_none()
    print("All error collector tests passed.")
