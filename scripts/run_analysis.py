"""Run from repo root: python scripts/run_analysis.py --bootstrap 20."""
import argparse
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.pipeline import run_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("Use at least two bootstrap repetitions")
    run_analysis(ROOT, args.bootstrap)
