"""Validation metrics used by model-risk teams. Pure functions, unit-tested."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    """Max distance between the score distributions of defaulters and non-defaulters."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    y = y[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected, actual, bins=10):
    """Population Stability Index on quantile bins of the expected (development) sample.
    Rule of thumb: < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, p, bins=10):
    """Observed vs predicted default rate by score decile, with a binomial z-score per bucket."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bucket"] = pd.qcut(df.p.rank(method="first"), bins, labels=False) + 1
    g = df.groupby("bucket").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    g["z"] = (g.observed - g.predicted) / np.sqrt(g.predicted * (1 - g.predicted) / g.n)
    return g


def rag(value, green, amber, higher_is_better=True):
    """Red / Amber / Green against documented thresholds."""
    if higher_is_better:
        return "GREEN" if value >= green else "AMBER" if value >= amber else "RED"
    return "GREEN" if value <= green else "AMBER" if value <= amber else "RED"
