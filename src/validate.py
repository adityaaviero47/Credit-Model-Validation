"""Independent validation of a credit-default (PD) model, in the style of a bank model-risk review.

Model under review: gradient-boosted PD model built on 36-month LendingClub loans issued 2012-2013.
Validation: in-time holdout (2012-13) and out-of-time sample (2014 loans the model never saw), covering
discrimination, calibration, stability, robustness, segment performance and benchmarking, each scored
Red / Amber / Green against thresholds written down before looking at the results.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance
from metrics import gini, ks, psi, calibration_table, rag

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAG_RULES = {  # set before validation, standard industry rules of thumb
    "oot_gini": (0.35, 0.25),            # >= 0.35 green, >= 0.25 amber, else red
    "gini_drop_pct": (10, 20),           # relative drop dev->OOT, lower is better
    "score_psi": (0.10, 0.25),           # lower is better
    "max_feature_csi": (0.10, 0.25),     # lower is better
    "calib_gap_pts": (1.0, 2.0),         # |mean predicted - observed| in % points, OOT
    "buckets_outside_2sd": (2, 4),       # deciles with |z| > 2 on OOT, lower is better
    "monotonic_violation_pct": (1.0, 5.0),
}
NUM = ["loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
       "revol_util", "total_acc", "mort_acc", "bc_util", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
       "pub_rec_bankruptcies", "acc_open_past_24mths", "avg_cur_bal", "tot_hi_cred_lim", "mths_since_last_delinq",
       "credit_history_months", "emp_years", "log_income", "loan_to_income"]
CAT = ["home_ownership", "verification_status", "purpose"]


def load():
    d = pd.read_csv(ROOT / "data" / "lc_36m_2012_2014.csv.gz", low_memory=False)
    d["issue_d"] = pd.to_datetime(d.issue_d); d["year"] = d.issue_d.dt.year
    d["default"] = (d.loan_status == "Charged Off").astype(int)
    for c in ["mort_acc", "num_tl_op_past_12m", "acc_open_past_24mths", "avg_cur_bal", "tot_hi_cred_lim", "mths_since_last_delinq"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["credit_history_months"] = ((d.issue_d - pd.to_datetime(d.earliest_cr_line, errors="coerce")).dt.days / 30.44)
    d["emp_years"] = d.emp_length.astype(str).str.extract(r"(\d+)")[0].astype(float)
    d.loc[d.emp_length.astype(str).str.startswith("<"), "emp_years"] = 0.5
    d["log_income"] = np.log1p(d.annual_inc)
    d["loan_to_income"] = d.loan_amnt / d.annual_inc.replace(0, np.nan)
    d = d[d.annual_inc > 0]
    return d


def design(d, cats=None):
    X = d[NUM].copy()
    for c in CAT:
        levels = cats[c] if cats else sorted(d[c].dropna().unique())
        for l in levels:
            X[f"{c}={l}"] = (d[c] == l).astype(int)
    return X, (cats or {c: sorted(d[c].dropna().unique()) for c in CAT})


def main():
    d = load()
    dev = d[d.year <= 2013]; oot = d[d.year == 2014]
    rng = np.random.default_rng(0)
    idx = rng.random(len(dev)) < 0.7
    tr, ho = dev[idx], dev[~idx]
    Xtr, cats = design(tr); Xho, _ = design(ho, cats); Xoot, _ = design(oot, cats)

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, random_state=0)
    model.fit(Xtr, tr.default)
    challenger = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    challenger.fit(Xtr, tr.default)
    p = {s: model.predict_proba(X)[:, 1] for s, X in [("train", Xtr), ("holdout", Xho), ("oot", Xoot)]}
    pc = {s: challenger.predict_proba(X)[:, 1] for s, X in [("holdout", Xho), ("oot", Xoot)]}
    y = {"train": tr.default.values, "holdout": ho.default.values, "oot": oot.default.values}
    grade_rank = {g: i for i, g in enumerate(sorted(d.sub_grade.unique()))}       # LendingClub's own grade as a benchmark
    pg = {s: x.sub_grade.map(grade_rank).values for s, x in [("holdout", ho), ("oot", oot)]}

    out = {"samples": {"train": len(tr), "holdout": len(ho), "oot": len(oot)},
           "default_rate": {s: round(float(v.mean()), 4) for s, v in y.items()},
           "oot_unresolved_excluded_pct": None}
    # 1 discrimination
    disc = {}
    for s in ["train", "holdout", "oot"]:
        disc[s] = {"auc": round(roc_auc_score(y[s], p[s]), 4), "gini": round(gini(y[s], p[s]), 4), "ks": round(ks(y[s], p[s]), 4)}
    out["discrimination"] = disc
    out["benchmarks_oot"] = {"model_gini": disc["oot"]["gini"],
                             "challenger_logistic_gini": round(gini(y["oot"], pc["oot"]), 4),
                             "lendingclub_subgrade_gini": round(gini(y["oot"], pg["oot"]), 4)}
    drop = 100 * (disc["holdout"]["gini"] - disc["oot"]["gini"]) / disc["holdout"]["gini"]
    # 2 calibration
    cal = calibration_table(y["oot"], p["oot"])
    cal.round(4).to_csv(RES / "calibration_oot.csv", index=False)
    gap = 100 * abs(p["oot"].mean() - y["oot"].mean())
    outside = int((cal.z.abs() > 2).sum())
    out["calibration_oot"] = {"mean_predicted": round(float(p["oot"].mean()), 4), "observed": round(float(y["oot"].mean()), 4),
                              "gap_pts": round(float(gap), 2), "deciles_outside_2sd": outside,
                              "brier": round(brier_score_loss(y["oot"], p["oot"]), 4)}
    # 3 stability
    score_psi = psi(p["holdout"], p["oot"])
    csi = {c: round(psi(Xtr[c].values, Xoot[c].values), 4) for c in NUM}
    csi = dict(sorted(csi.items(), key=lambda kv: -kv[1]))
    out["stability"] = {"score_psi": round(score_psi, 4), "top_feature_csi": dict(list(csi.items())[:5])}
    # 4 robustness: perturbations on the OOT sample
    def shift(col, f):
        X = Xoot.copy(); X[col] = X[col] * f
        if col == "annual_inc":
            X["log_income"] = np.log1p(X.annual_inc); X["loan_to_income"] = X.loan_amnt / X.annual_inc
        return model.predict_proba(X)[:, 1]
    base = p["oot"]
    rob = {}
    for col, f, want in [("dti", 1.2, "up"), ("annual_inc", 0.8, "up"), ("inq_last_6mths", 2.0, "up"), ("revol_util", 1.2, "up")]:
        q = shift(col, f); diff = q - base
        viol = float(((diff < -1e-9) if want == "up" else (diff > 1e-9)).mean() * 100)
        rob[f"{col} x{f}"] = {"mean_pd_change_pts": round(float(diff.mean() * 100), 3), "wrong_direction_pct": round(viol, 2)}
    Xm = Xoot.copy(); Xm.loc[rng.random(len(Xm)) < 0.1, "revol_util"] = np.nan
    rob["10% revol_util missing"] = {"gini": round(gini(y["oot"], model.predict_proba(Xm)[:, 1]), 4)}
    out["robustness_oot"] = rob
    worst_viol = max(v.get("wrong_direction_pct", 0) for v in rob.values())
    # 5 segments
    seg = {}
    for col in ["purpose", "home_ownership", "verification_status"]:
        for lvl, g in oot.groupby(col):
            if len(g) >= 2000 and g.default.nunique() == 2:
                m = oot[col].values == lvl
                seg[f"{col}={lvl}"] = {"n": int(m.sum()), "gini": round(gini(y["oot"][m], p["oot"][m]), 4),
                                       "observed": round(float(y["oot"][m].mean()), 4), "predicted": round(float(p["oot"][m].mean()), 4)}
    out["segments_oot"] = seg
    weakest = min(seg.items(), key=lambda kv: kv[1]["gini"])
    # 6 explainability: permutation importance on a holdout sample
    sm = rng.choice(len(Xho), size=min(20000, len(Xho)), replace=False)
    pi = permutation_importance(model, Xho.iloc[sm], y["holdout"][sm], scoring="roc_auc", n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=Xho.columns).sort_values(ascending=False)
    out["top_drivers"] = {k: round(float(v), 4) for k, v in imp.head(6).items()}
    # RAG summary
    out["rag"] = {
        "OOT discrimination (Gini)": [disc["oot"]["gini"], rag(disc["oot"]["gini"], *RAG_RULES["oot_gini"])],
        "Gini drop holdout->OOT (%)": [round(drop, 1), rag(drop, *RAG_RULES["gini_drop_pct"], higher_is_better=False)],
        "Score PSI holdout->OOT": [round(score_psi, 3), rag(score_psi, *RAG_RULES["score_psi"], higher_is_better=False)],
        "Max feature CSI": [list(csi.values())[0], rag(list(csi.values())[0], *RAG_RULES["max_feature_csi"], higher_is_better=False)],
        "OOT calibration gap (pts)": [round(gap, 2), rag(gap, *RAG_RULES["calib_gap_pts"], higher_is_better=False)],
        "Deciles outside 2 s.d.": [outside, rag(outside, *RAG_RULES["buckets_outside_2sd"], higher_is_better=False)],
        "Monotonicity violations (%)": [round(worst_viol, 2), rag(worst_viol, *RAG_RULES["monotonic_violation_pct"], higher_is_better=False)],
    }
    out["weakest_segment"] = [weakest[0], weakest[1]["gini"]]
    out["overfit_gini_drop_train_to_holdout_pct"] = round(100 * (disc["train"]["gini"] - disc["holdout"]["gini"]) / disc["train"]["gini"], 1)

    # 7 remediation the validator recommends, tested the same way
    # (a) monotonic constraints on risk drivers whose direction is known
    direction = {"dti": 1, "annual_inc": -1, "log_income": -1, "inq_last_6mths": 1, "revol_util": 1, "loan_to_income": 1,
                 "bc_util": 1, "delinq_2yrs": 1, "pub_rec_bankruptcies": 1}
    cst = [direction.get(c, 0) for c in Xtr.columns]
    mono = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, monotonic_cst=cst, random_state=0).fit(Xtr, tr.default)
    pm = mono.predict_proba(Xoot)[:, 1]
    def shift_m(col, f):
        X = Xoot.copy(); X[col] = X[col] * f
        if col == "annual_inc":
            X["log_income"] = np.log1p(X.annual_inc); X["loan_to_income"] = X.loan_amnt / X.annual_inc
        return mono.predict_proba(X)[:, 1]
    mviol = max(float((((shift_m(c, f) - pm) < -1e-9).mean()) * 100) for c, f in [("dti", 1.2), ("annual_inc", 0.8), ("inq_last_6mths", 2.0), ("revol_util", 1.2)])
    # (b) periodic recalibration: fit an intercept-and-slope (Platt) adjustment on Jan-Jun 2014 outcomes, test on Jul-Dec 2014
    h1 = (oot.issue_d.dt.month <= 6).values
    logit = lambda q: np.log(q / (1 - q))
    lr = LogisticRegression().fit(logit(pm[h1]).reshape(-1, 1), y["oot"][h1])
    pr = lr.predict_proba(logit(pm[~h1]).reshape(-1, 1))[:, 1]
    cal2 = calibration_table(y["oot"][~h1], pr)
    out["remediation"] = {
        "monotonic_model": {"oot_gini": round(gini(y["oot"], pm), 4), "monotonicity_violation_pct": round(mviol, 2)},
        "recalibrated_h2_2014": {"n": int((~h1).sum()), "mean_predicted": round(float(pr.mean()), 4),
                                 "observed": round(float(y["oot"][~h1].mean()), 4),
                                 "gap_pts": round(float(100 * abs(pr.mean() - y["oot"][~h1].mean())), 2),
                                 "deciles_outside_2sd": int((cal2.z.abs() > 2).sum()),
                                 "before_gap_pts_same_sample": round(float(100 * abs(pm[~h1].mean() - y["oot"][~h1].mean())), 2),
                                 "before_deciles_outside_2sd": int((calibration_table(y["oot"][~h1], pm[~h1]).z.abs() > 2).sum())}}
    out["rag_rules"] = RAG_RULES
    json.dump(out, open(RES / "validation.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
