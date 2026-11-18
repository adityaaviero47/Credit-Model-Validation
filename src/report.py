"""Write the validation report (results/validation_report.md) from results/validation.json."""
import json
from pathlib import Path
R = json.load(open(Path(__file__).resolve().parents[1] / "results" / "validation.json"))
d, c, s, rm = R["discrimination"], R["calibration_oot"], R["stability"], R["remediation"]
L = ["# Model Validation Report: 36-month Unsecured Personal Loan PD Model", "",
     f"Development: 2012-13 loans ({R['samples']['train']:,} train / {R['samples']['holdout']:,} holdout). "
     f"Out-of-time: {R['samples']['oot']:,} loans issued in 2014.", "",
     "## Summary of tests", "", "| Test | Result | Rating |", "|---|---|---|"]
for k, (v, r) in R["rag"].items():
    L.append(f"| {k} | {v} | **{r}** |")
L += ["", "## Findings", "",
      f"1. **Discrimination holds out of time.** OOT Gini {d['oot']['gini']:.3f} vs holdout {d['holdout']['gini']:.3f}; beats a logistic "
      f"challenger ({R['benchmarks_oot']['challenger_logistic_gini']:.3f}) and LendingClub's own sub-grade ({R['benchmarks_oot']['lendingclub_subgrade_gini']:.3f}).",
      f"2. **Over-fitting:** train Gini {d['train']['gini']:.3f} falls {R['overfit_gini_drop_train_to_holdout_pct']}% on the holdout. Holdout and OOT agree, "
      "so out-of-sample performance is reliable, but the developer should regularise further.",
      f"3. **Population is stable (score PSI {s['score_psi']:.3f}), but the model under-predicts risk:** {c['mean_predicted']*100:.1f}% predicted vs "
      f"{c['observed']*100:.1f}% observed, with {c['deciles_outside_2sd']} of 10 deciles outside 2 s.d. The default rate rose while the borrower mix did not, "
      "so the remedy is recalibration, not a rebuild.",
      f"4. **Not monotonic:** raising DTI by 20% lowers predicted PD for {R['robustness_oot']['dti x1.2']['wrong_direction_pct']}% of loans, which "
      "cannot be justified to credit committees or regulators.",
      f"5. **Weakest segment:** {R['weakest_segment'][0]} (Gini {R['weakest_segment'][1]:.3f}).", "",
      "## Remediation tested", "",
      f"- Monotonic constraints on 9 risk drivers: violations {rm['monotonic_model']['monotonicity_violation_pct']}%, OOT Gini {rm['monotonic_model']['oot_gini']:.3f} (no loss).",
      f"- Platt recalibration on Jan-Jun 2014, tested on Jul-Dec 2014: gap {rm['recalibrated_h2_2014']['before_gap_pts_same_sample']} -> "
      f"{rm['recalibrated_h2_2014']['gap_pts']} pts. Defaults kept rising through 2014, so recommend quarterly recalibration with a PSI and "
      "calibration-gap monitoring trigger.", "",
      "## Conclusion", "", "**Approve with conditions:** deploy the monotonic version, recalibrate before go-live and quarterly after, "
      "and monitor calibration gap and PSI monthly against the thresholds above."]
(Path(__file__).resolve().parents[1] / "results" / "validation_report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
