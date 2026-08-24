"""Run from the repository root: python scripts/prepare_data.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.data import prepare_data

if __name__ == "__main__":
    data, audit = prepare_data(ROOT)
    print(audit.to_string(index=False))
    print(f"Clean customers: {data.customer_id.nunique():,}")
