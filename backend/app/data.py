import csv
import math
import random
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "coffee_raw.csv"


NUMERIC_FEATURES = {
    "Aroma": "aroma",
    "Flavor": "flavor",
    "Aftertaste": "aftertaste",
    "Acidity": "acidity",
    "Body": "body",
    "Balance": "balance",
    "Uniformity": "uniformity",
    "Clean.Cup": "clean_cup",
    "Sweetness": "sweetness",
    "Moisture": "moisture",
    "Category.Two.Defects": "defects",
    "altitude_mean_meters": "altitude_m",
}
CATEGORICAL_FEATURES = {
    "Country.of.Origin": "country",
}
TARGET_RAW = "Processing.Method"
TARGET = "processing_method"

PROCESSING_MAP = {
    "Washed / Wet": "Washed",
    "Natural / Dry": "Natural",
    "Pulped natural / honey": "Honey",
    "Semi-washed / Semi-pulped": "Semi-Washed",
}

TOP_COUNTRIES = [
    "Ethiopia", "Mexico", "Colombia", "Guatemala", "Brazil",
    "Taiwan", "United States (Hawaii)", "Honduras", "Costa Rica",
    "Tanzania, United Republic Of", "Kenya", "Uganda",
]

FEATURE_ORDER = list(NUMERIC_FEATURES.values()) + list(CATEGORICAL_FEATURES.values())
FEATURE_TYPES = {v: "numeric" for v in NUMERIC_FEATURES.values()}
FEATURE_TYPES.update({v: "categorical" for v in CATEGORICAL_FEATURES.values()})


def _to_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_records():
    """Read the raw CSV and return a list of clean records (dicts)."""
    records = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_method = (row.get(TARGET_RAW) or "").strip()
            label = PROCESSING_MAP.get(raw_method)
            if label is None:
                continue  # drop "Other" / missing processing method

            rec = {}
            ok = True
            for raw_col, clean_name in NUMERIC_FEATURES.items():
                val = _to_float(row.get(raw_col))
                if val is None:
                    ok = False
                    break
                rec[clean_name] = val
            if not ok:
                continue

            country = (row.get("Country.of.Origin") or "").strip()
            if not country:
                continue
            rec["country"] = country if country in TOP_COUNTRIES else "Other"

            # sanity bounds - drop obviously bad rows (0 altitude, absurd altitude, etc.)
            if not (0 < rec["altitude_m"] < 5000):
                continue
            if not (7.0 <= rec["moisture"] * 0 + rec["moisture"] <= 100.0):
                pass  # moisture already a %, no-op guard kept for clarity

            rec[TARGET] = label
            records.append(rec)
    return records


def train_test_split(records, test_size=0.25, seed=42):
    rng = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n_test = int(len(data) * test_size)
    return data[n_test:], data[:n_test]


def class_balance(records):
    counts = {}
    for r in records:
        counts[r[TARGET]] = counts.get(r[TARGET], 0) + 1
    return counts


def dataset_summary():
    records = load_records()
    counts = class_balance(records)
    countries = sorted({r["country"] for r in records})
    ranges = {}
    for f in NUMERIC_FEATURES.values():
        vals = [r[f] for r in records]
        ranges[f] = {"min": round(min(vals), 2), "max": round(max(vals), 2)}
    return {
        "n_samples": len(records),
        "class_counts": counts,
        "features": FEATURE_ORDER,
        "feature_types": FEATURE_TYPES,
        "numeric_ranges": ranges,
        "countries": countries,
        "sample_rows": records[:8],
    }


if __name__ == "__main__":
    s = dataset_summary()
    print(s["n_samples"], s["class_counts"])
