# Model Validation Report: 36-month Unsecured Personal Loan PD Model

Development: 2012-13 loans (100,820 train / 43,072 holdout). Out-of-time: 152,692 loans issued in 2014.

## Summary of tests

| Test | Result | Rating |
|---|---|---|
| OOT discrimination (Gini) | 0.345 | **AMBER** |
| Gini drop holdout->OOT (%) | -3.1 | **GREEN** |
| Score PSI holdout->OOT | 0.003 | **GREEN** |
| Max feature CSI | 0.0563 | **GREEN** |
| OOT calibration gap (pts) | 1.57 | **AMBER** |
| Deciles outside 2 s.d. | 9 | **RED** |
| Monotonicity violations (%) | 10.73 | **RED** |

## Findings

1. **Discrimination holds out of time.** OOT Gini 0.345 vs holdout 0.335; beats a logistic challenger (0.326) and LendingClub's own sub-grade (0.320).
2. **Over-fitting:** train Gini 0.459 falls 27.0% on the holdout. Holdout and OOT agree, so out-of-sample performance is reliable, but the developer should regularise further.
3. **Population is stable (score PSI 0.003), but the model under-predicts risk:** 12.7% predicted vs 14.3% observed, with 9 of 10 deciles outside 2 s.d. The default rate rose while the borrower mix did not, so the remedy is recalibration, not a rebuild.
4. **Not monotonic:** raising DTI by 20% lowers predicted PD for 10.73% of loans, which cannot be justified to credit committees or regulators.
5. **Weakest segment:** purpose=other (Gini 0.298).

## Remediation tested

- Monotonic constraints on 9 risk drivers: violations 0.0%, OOT Gini 0.344 (no loss).
- Platt recalibration on Jan-Jun 2014, tested on Jul-Dec 2014: gap 2.4 -> 1.66 pts. Defaults kept rising through 2014, so recommend quarterly recalibration with a PSI and calibration-gap monitoring trigger.

## Conclusion

**Approve with conditions:** deploy the monotonic version, recalibrate before go-live and quarterly after, and monitor calibration gap and PSI monthly against the thresholds above.
