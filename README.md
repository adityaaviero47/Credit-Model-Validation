# Credit Risk Model Validation: Independent Review of a PD Model

![ci](https://github.com/adityaaviero47/Credit-Model-Validation/actions/workflows/ci.yml/badge.svg)

**What this is:** an independent validation of a probability-of-default (PD) model for unsecured personal loans, done the way a bank's model-risk team reviews a model before it's used. The developer's model gets the full battery of tests, each test is rated **Red / Amber / Green** against thresholds fixed in advance, the recommended fixes are tried and measured, and the report is generated automatically.

**The data:** 296,584 real 36-month LendingClub loans with final outcomes (fully paid or charged off), split by when they were issued:
- **Development:** 2012–2013 loans (144k), divided 70/30 into train and holdout.
- **Out-of-time (OOT):** **2014 loans (153k)** that the model never saw.

Only information known at application time is used. Interest rate, grade and every post-loan payment field are excluded, and a test enforces this. LendingClub's own sub-grade is kept only as a benchmark to compare against.

## Results (`results/validation_report.md`)

| Test | Result | Rating |
|---|---|---|
| OOT discrimination (Gini) | 0.345 | AMBER |
| Gini change, holdout → OOT | +3.1% (no decay) | GREEN |
| Score stability (PSI), holdout → OOT | 0.003 | GREEN |
| Largest feature shift (CSI) | 0.056 | GREEN |
| OOT calibration gap | 1.6 pts | AMBER |
| Score deciles outside ±2 s.d. | 9 of 10 | **RED** |
| Monotonicity violations | 10.7% | **RED** |

**Findings**
1. **The ranking of risky borrowers holds up on future loans.** OOT Gini is 0.345, the same as the holdout, and the model beats both a logistic-regression challenger (0.326) and LendingClub's own sub-grade (0.320).
2. **Over-fitting:** train Gini 0.459 falls 27% on the holdout. The developer should regularise further.
3. **The borrower mix is stable, but the model under-predicts defaults.** It predicts 12.7% against 14.3% observed in 2014, and 9 of 10 deciles are under-predicted. Defaults rose while the population didn't change (PSI 0.003), so the remedy is **recalibration, not a rebuild**.
4. **It isn't monotonic.** Raising a borrower's debt-to-income by 20% *lowers* their predicted PD for 10.7% of loans, which a credit committee can't accept.
5. **The weakest segment** is loans with purpose "other" (Gini 0.298).

**Remediation, tested the same way**
- **Monotonic constraints** on 9 risk drivers bring violations to **0%**, with no loss of discrimination (OOT Gini 0.344).
- **Platt recalibration**, fitted on Jan–Jun 2014 and tested on Jul–Dec 2014, cuts the calibration gap from **2.4 to 1.7 points**. Defaults kept rising through 2014, so a one-off recalibration isn't enough: **quarterly recalibration with PSI and calibration-gap monitoring triggers** is recommended.

**Conclusion: approve with conditions.**

## Tests run (`src/validate.py`)
- **Discrimination:** AUC, Gini and KS on train, holdout and OOT, plus benchmarks against a challenger model and the lender's own grade.
- **Calibration:** predicted vs observed default rate per decile, with a binomial z-score for each, and the Brier score.
- **Stability:** score PSI and a characteristic stability index (CSI) for every input.
- **Robustness:** stress-shifts to DTI, income, inquiries and utilisation, with checks on the direction of the response, and 10% missing data.
- **Segments:** Gini and calibration by loan purpose, home ownership and income verification.
- **Explainability:** permutation importance of the top drivers.

## Run it
```bash
pip install -r requirements.txt
python src/validate.py      # results/validation.json, results/calibration_oot.csv
python src/report.py        # results/validation_report.md
python -m pytest -q         # metric unit tests, out-of-time split and no-leakage checks
```
`src/extract_data.py` rebuilds `data/` from the public LendingClub files (github.com/kuhnrl30/LendingClubData).

## Limits
- **Resolved loans only:** about 6% of 2014 loans were still active when the data was published, so they're excluded. That can slightly understate 2014 defaults.
- **No macroeconomic variables:** the model has none, so a full review would add a stress-testing scenario.
- **Public data:** this is one lender's public data, not a bank's portfolio. The method is standard; the thresholds are common rules of thumb, not one bank's policy.
