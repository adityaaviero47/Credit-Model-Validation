"""Rebuild data/lc_36m_2012_2014.csv.gz from the public LendingClub loan files.
Source: github.com/kuhnrl30/LendingClubData (commit b9203b8), yearly .rda files of LendingClub's published loan data.
Usage: git clone https://github.com/kuhnrl30/LendingClubData && pip install pyreadr && python src/extract_data.py LendingClubData
"""
import sys
from pathlib import Path
import pandas as pd
import pyreadr

KEEP = ["id", "issue_d", "term", "loan_status", "loan_amnt", "grade", "sub_grade", "int_rate", "emp_length", "home_ownership",
        "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "bc_util", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
        "pub_rec_bankruptcies", "acc_open_past_24mths", "avg_cur_bal", "tot_hi_cred_lim", "mths_since_last_delinq"]
src = Path(sys.argv[1])
frames = []
for y in (2012, 2013, 2014):
    d = list(pyreadr.read_r(str(src / "data" / f"LendingClubData_{y}.rda")).values())[0]
    frames.append(d[d.term.astype(str).str.strip() == "36"][KEEP])
d = pd.concat(frames)
d = d[d.loan_status.isin(["Fully Paid", "Charged Off"])]          # resolved loans only
out = Path(__file__).resolve().parents[1] / "data" / "lc_36m_2012_2014.csv.gz"
d.to_csv(out, index=False, compression="gzip")
print(out, d.shape)
