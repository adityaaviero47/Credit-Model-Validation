import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from validate import load, NUM


def test_out_of_time_split_and_no_leakage_features():
    d = load()
    assert set(d.year.unique()) == {2012, 2013, 2014}
    assert (d.term.astype(str).str.strip() == "36").all()
    assert set(d.loan_status.unique()) == {"Fully Paid", "Charged Off"}
    banned = {"int_rate", "grade", "sub_grade", "total_pymnt", "recoveries", "last_pymnt_amnt", "out_prncp", "installment"}
    assert not banned & set(NUM)                                   # no post-origination or pricing fields as inputs
