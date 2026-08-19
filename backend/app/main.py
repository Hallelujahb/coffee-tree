
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from .data import (
    load_records, train_test_split, dataset_summary,
    FEATURE_ORDER, FEATURE_TYPES, TARGET, NUMERIC_FEATURES, TOP_COUNTRIES,
)
from .decision_tree import DecisionTree

app = FastAPI(title="Cupping Lab - Decision Tree API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory state: the app is a single-session teaching demo, not multi-tenant
_state = {
    "tree": None,
    "train": None,
    "test": None,
}


def _ensure_data():
    if _state["train"] is None:
        records = load_records()
        train, test = train_test_split(records, test_size=0.25, seed=42)
        _state["train"], _state["test"] = train, test
    return _state["train"], _state["test"]


class TrainParams(BaseModel):
    criterion: str = Field(default="entropy", pattern="^(entropy|gini)$")
    max_depth: int = Field(default=4, ge=1, le=10)
    min_samples_split: int = Field(default=10, ge=2, le=200)
    min_samples_leaf: int = Field(default=4, ge=1, le=100)
    min_impurity_decrease: float = Field(default=0.001, ge=0.0, le=0.5)


class PredictInput(BaseModel):
    aroma: float
    flavor: float
    aftertaste: float
    acidity: float
    body: float
    balance: float
    uniformity: float
    clean_cup: float
    sweetness: float
    moisture: float
    defects: float
    altitude_m: float
    country: str


@app.get("/api/dataset/summary")
def get_dataset_summary():
    return dataset_summary()


@app.post("/api/train")
def train_model(params: TrainParams):
    train, test = _ensure_data()
    tree = DecisionTree(
        criterion=params.criterion,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        min_impurity_decrease=params.min_impurity_decrease,
    )
    tree.fit(train, FEATURE_ORDER, FEATURE_TYPES, TARGET)
    _state["tree"] = tree

    train_metrics = tree.evaluate(train)
    test_metrics = tree.evaluate(test)

    # majority-class baseline, for an honest "did the tree actually learn anything" comparison
    from .decision_tree import counts_of, accuracy_of_counts
    baseline_acc = accuracy_of_counts(counts_of(train, TARGET))

    return {
        "params": params.model_dump(),
        "tree": tree.to_json(),
        "depth": tree.depth(),
        "leaf_count": tree.leaf_count(),
        "feature_importances": tree.feature_importances(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_accuracy": round(baseline_acc, 4),
        "classes": tree.classes_,
    }


@app.get("/api/tree")
def get_tree():
    if _state["tree"] is None:
        raise HTTPException(status_code=404, detail="No tree trained yet. POST /api/train first.")
    return {"tree": _state["tree"].to_json(), "classes": _state["tree"].classes_}


@app.post("/api/predict")
def predict(sample: PredictInput):
    if _state["tree"] is None:
        raise HTTPException(status_code=404, detail="No tree trained yet. POST /api/train first.")
    record = sample.model_dump()
    result = _state["tree"].predict_one(record)
    return result


@app.get("/api/health")
def health():
    return {"status": "ok"}
