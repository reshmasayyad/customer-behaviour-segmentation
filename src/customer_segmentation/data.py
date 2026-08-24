"""Download the official workbook and audit transaction cleaning."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SOURCE_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
REQUIRED = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def obtain_workbook(raw_dir):
    """Download once, verify the ZIP, and extract only the expected workbook."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    workbook = raw_dir / "online_retail_II.xlsx"
    if workbook.exists():
        return workbook
    archive = raw_dir / "online_retail_ii.zip"
    if not archive.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            archive.write_bytes(response.read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Official archive checksum changed; inspect the dataset before proceeding.")
    with zipfile.ZipFile(archive) as zipped:
        workbook.write_bytes(zipped.read("online_retail_II.xlsx"))
    return workbook


def load_transactions(workbook):
    """Read every worksheet; do not silently use only the first year."""
    sheets = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    frames = []
    for name, frame in sheets.items():
        missing = set(REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")
        frames.append(frame.assign(source_sheet=name))
    return pd.concat(frames, ignore_index=True)


def clean_transactions(raw):
    """Create UK positive merchandise purchases and a sequential audit.

    Returns/cancellations are excluded, not subtracted: spending is gross positive
    merchandise spending. Numeric five-digit stock codes with optional letter
    suffixes identify merchandise; service/adjustment codes are excluded.
    Exact duplicate rows are removed (a documented, revisable assumption).
    """
    frame = raw.copy()
    audit = [{"step": "raw_all_sheets", "removed": 0, "remaining": len(frame)}]

    def retain(mask, step):
        nonlocal frame
        before = len(frame)
        frame = frame.loc[mask].copy()
        audit.append({"step": step, "removed": before - len(frame), "remaining": len(frame)})

    retain(~frame.duplicated(subset=REQUIRED), "remove_exact_duplicates")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    customer = pd.to_numeric(frame["Customer ID"], errors="coerce")
    retain(customer.notna() & customer.mod(1).eq(0), "require_valid_customer_id")
    frame["customer_id"] = frame["Customer ID"].astype("int64").astype(str)
    retain(frame["InvoiceDate"].notna() & frame["Price"].notna() & frame["Quantity"].notna(), "require_valid_date_price_quantity")
    frame["invoice_id"] = frame["Invoice"].astype(str).str.strip()
    retain(~frame["invoice_id"].str.upper().str.startswith("C"), "exclude_cancellation_invoices")
    retain(frame["Quantity"].gt(0) & frame["Price"].gt(0), "require_positive_quantity_price")
    frame["stock_code"] = frame["StockCode"].astype(str).str.strip()
    retain(frame["stock_code"].str.fullmatch(r"\d{5}[A-Za-z]*", na=False), "require_merchandise_stock_code")
    retain(frame["Country"].eq("United Kingdom"), "restrict_to_uk_customers")
    frame["revenue"] = frame["Quantity"] * frame["Price"]
    frame = frame.sort_values("InvoiceDate").reset_index(drop=True)
    return frame, pd.DataFrame(audit)


def prepare_data(root):
    """Cache cleaned rows locally; always retain the original source workbook."""
    root = Path(root)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    root.joinpath("results").mkdir(exist_ok=True)
    cache = processed / "uk_purchases.pkl"
    audit_path = root / "results/cleaning_audit.csv"
    if cache.exists() and audit_path.exists():
        return pd.read_pickle(cache), pd.read_csv(audit_path)
    workbook = obtain_workbook(root / "data/raw")
    raw = load_transactions(workbook)
    clean, audit = clean_transactions(raw)
    clean.to_pickle(cache)
    audit.to_csv(audit_path, index=False)
    manifest = {
        "source_url": SOURCE_URL, "zip_sha256": SOURCE_SHA256,
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "clean_rows": len(clean),
        "raw_start": str(raw.InvoiceDate.min()), "raw_end": str(raw.InvoiceDate.max()),
        "source_sheets": sorted(raw.source_sheet.unique().tolist()),
        "cleaning_scope": "UK positive merchandise purchases; gross spending excluding returns",
    }
    (root / "results/data_manifest.json").write_text(json.dumps(manifest, indent=2))
    return clean, audit
