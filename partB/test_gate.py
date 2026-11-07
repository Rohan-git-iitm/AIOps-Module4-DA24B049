"""test_gate.py -- tests of the pure decide() function.  Run from the submission root:  pytest partB -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eta_mlflow import decide  # noqa: E402

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 2.0}


def cand(**kw):
    return {**{"mae": 8.0, "mae_rush": 10.0, "mae_rain": 9.0, "p95_ms": 3.0}, **kw}


def test_clear_win_promotes():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert all(c["pass"] for c in out["criteria"].values())


def test_gain_just_inside_threshold():
    assert decide(PROD, cand(mae=9.69))["promote"] is True          # 3.1 % gain


def test_gain_just_outside_threshold():
    out = decide(PROD, cand(mae=9.71))                                # 2.9 % gain
    assert out["promote"] is False and out["criteria"]["mae_rel_gain"]["pass"] is False


def test_gain_exactly_on_threshold_passes():
    assert decide(PROD, cand(mae=9.7))["promote"] is True            # exactly 3 %


def test_worse_candidate_rejected():
    assert decide(PROD, cand(mae=11.0))["promote"] is False


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=9.99))["promote"] is True
    assert decide(PROD, cand(p95_ms=10.0))["promote"] is True
    out = decide(PROD, cand(p95_ms=10.01))
    assert out["promote"] is False and out["criteria"]["p95_ms"]["pass"] is False


def test_rush_slice_regression_blocks_despite_overall_gain():
    assert decide(PROD, cand(mae_rush=12.59))["promote"] is True     # +4.9 %
    out = decide(PROD, cand(mae_rush=12.61))                          # +5.1 %
    assert out["promote"] is False and out["criteria"]["rush_regression"]["pass"] is False


def test_rain_slice_regression_blocks_despite_overall_gain():
    assert decide(PROD, cand(mae_rain=11.54))["promote"] is True     # +4.9 %
    out = decide(PROD, cand(mae_rain=11.56))                          # +5.1 %
    assert out["promote"] is False and out["criteria"]["rain_regression"]["pass"] is False


def test_custom_thresholds():
    assert decide(PROD, cand(mae=5.0), min_rel_gain=0.60)["promote"] is False   # 50 % < 60 %
    assert decide(PROD, cand(mae=3.9), min_rel_gain=0.60)["promote"] is True    # 61 %
    assert decide(PROD, cand(p95_ms=15), max_p95_ms=20)["promote"] is True


def test_is_pure():
    p, c = dict(PROD), cand()
    first = decide(p, c)
    assert p == PROD and c == cand()                                  # inputs not mutated
    assert decide(p, c) == first                                      # deterministic
