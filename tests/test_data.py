import pandas as pd
from customer_segmentation.data import clean_transactions, REQUIRED


def test_returns_services_missing_ids_and_duplicates_are_audited():
    rows = [
        ["1", "12345", "item", 2, "2010-01-01", 3, 10, "United Kingdom"],
        ["1", "12345", "item", 2, "2010-01-01", 3, 10, "United Kingdom"],
        ["C2", "12345", "item", -1, "2010-01-02", 3, 10, "United Kingdom"],
        ["3", "POST", "postage", 1, "2010-01-03", 3, 10, "United Kingdom"],
        ["4", "12345", "item", 1, "2010-01-04", 3, None, "United Kingdom"],
        ["5", "12345", "item", 1, "2010-01-05", 3, 11, "France"],
    ]
    clean, audit = clean_transactions(pd.DataFrame(rows, columns=REQUIRED))
    assert len(clean) == 1
    assert clean.iloc[0].revenue == 6
    assert clean.iloc[0].customer_id == "10"
    assert audit.removed.sum() + len(clean) == len(rows)
