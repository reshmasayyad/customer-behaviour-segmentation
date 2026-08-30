"""Assign existing-model segments to a customer-feature CSV.

Input requires customer_id plus the four features in the documented units and
182-day observation window. Run after training; no model refitting occurs.
"""
from pathlib import Path
import argparse
import sys
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.models import transform


def score(input_path, output_path, model_path):
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_path, dtype={"customer_id": str}).set_index("customer_id")
    if not frame.index.is_unique:
        raise ValueError("Each customer_id must appear exactly once")
    if "window_days" not in frame or not frame.window_days.eq(artifact["window_days"]).all():
        raise ValueError(f"Input must include window_days={artifact['window_days']} for every customer")
    scaled = transform(artifact, frame)
    probabilities = artifact["model"].predict_proba(scaled)
    labels = artifact["model"].predict(scaled)
    output = pd.DataFrame({"customer_id": frame.index,
                           "segment": [artifact["segment_names"][int(label)] for label in labels],
                           "membership_confidence": probabilities.max(axis=1)})
    for component in range(probabilities.shape[1]):
        output[f"probability_{artifact['segment_names'][component]}"] = probabilities[:, component]
    output.to_csv(output_path, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/segmentation.joblib")
    args = parser.parse_args()
    print(score(args.input_csv, args.output_csv, args.model).to_string(index=False))
