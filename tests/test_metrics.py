import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from metrics import gini, ks, psi, calibration_table, rag


def test_perfect_and_random_ranking():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.1, .2, .3, .7, .8, .9])
    assert gini(y, p) == 1.0 and ks(y, p) == 1.0
    rng = np.random.default_rng(0); y2 = rng.integers(0, 2, 20000); p2 = rng.random(20000)
    assert abs(gini(y2, p2)) < 0.05 and ks(y2, p2) < 0.05


def test_psi_zero_for_same_population_and_large_for_shift():
    rng = np.random.default_rng(1); a = rng.normal(size=50000)
    assert psi(a, rng.normal(size=50000)) < 0.01
    assert psi(a, rng.normal(loc=1.0, size=50000)) > 0.25


def test_calibration_flags_underprediction():
    rng = np.random.default_rng(2); p = rng.uniform(0.05, 0.3, 50000)
    y = (rng.random(50000) < p * 1.3).astype(int)             # true risk 30% higher than predicted
    t = calibration_table(y, p)
    assert (t.observed > t.predicted).all() and (t.z > 2).sum() >= 8


def test_rag_directions():
    assert rag(0.40, 0.35, 0.25) == "GREEN" and rag(0.30, 0.35, 0.25) == "AMBER" and rag(0.2, 0.35, 0.25) == "RED"
    assert rag(0.05, 0.10, 0.25, higher_is_better=False) == "GREEN" and rag(0.3, 0.10, 0.25, higher_is_better=False) == "RED"
