
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


def gini_impurity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


def entropy_impurity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def accuracy_of_counts(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def impurity(counts: dict, criterion: str) -> float:
    return gini_impurity(counts) if criterion == "gini" else entropy_impurity(counts)


def counts_of(records, target_key) -> dict:
    counts = {}
    for r in records:
        c = r[target_key]
        counts[c] = counts.get(c, 0) + 1
    return counts


_id_counter = {"n": 0}


def _next_id():
    _id_counter["n"] += 1
    return _id_counter["n"]


@dataclass
class Node:
    depth: int
    records: list = field(repr=False)
    target_key: str
    criterion: str
    id: int = field(default_factory=_next_id)
    is_leaf: bool = True
    prediction: Optional[str] = None
    class_counts: dict = field(default_factory=dict)
    n_samples: int = 0
    gini: float = 0.0
    entropy: float = 0.0
    accuracy: float = 0.0
    feature: Optional[str] = None
    feature_type: Optional[str] = None      # "numeric" | "categorical"
    threshold: Optional[float] = None       # numeric split point
    category: Optional[str] = None          # categorical split value
    gain: float = 0.0                       # impurity reduction achieved by this split
    yes: Optional["Node"] = None
    no: Optional["Node"] = None

    def __post_init__(self):
        self.class_counts = counts_of(self.records, self.target_key)
        self.n_samples = len(self.records)
        self.gini = gini_impurity(self.class_counts)
        self.entropy = entropy_impurity(self.class_counts)
        self.accuracy = accuracy_of_counts(self.class_counts)
        if self.class_counts:
            self.prediction = max(self.class_counts, key=self.class_counts.get)


class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=4, min_samples_split=10,
                 min_samples_leaf=4, min_impurity_decrease=0.001):
        assert criterion in ("gini", "entropy")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[Node] = None
        self.feature_names: list[str] = []
        self.feature_types: dict[str, str] = {}
        self.target_key = None
        self.classes_: list[str] = []
        self._importance_accum: dict[str, float] = {}

    # ---------- fitting ----------

    def fit(self, records, feature_names, feature_types, target_key):
        _id_counter["n"] = 0  # reset ids for a fresh, readable tree
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.target_key = target_key
        self.classes_ = sorted({r[target_key] for r in records})
        self._importance_accum = {f: 0.0 for f in feature_names}
        self.root = self._build(records, depth=0)
        return self

    def _build(self, records, depth) -> Node:
        node = Node(depth=depth, records=records, target_key=self.target_key,
                    criterion=self.criterion)

        # stopping conditions
        if node.gini == 0.0:
            return node  # pure leaf, nothing to gain
        if depth >= self.max_depth:
            return node
        if node.n_samples < self.min_samples_split:
            return node

        best = self._best_split(records, node)
        if best is None or best["gain"] < self.min_impurity_decrease:
            return node

        node.is_leaf = False
        node.feature = best["feature"]
        node.feature_type = best["feature_type"]
        node.threshold = best.get("threshold")
        node.category = best.get("category")
        node.gain = best["gain"]

        self._importance_accum[node.feature] = (
            self._importance_accum.get(node.feature, 0.0) + best["gain"] * node.n_samples
        )

        node.yes = self._build(best["yes_records"], depth + 1)
        node.no = self._build(best["no_records"], depth + 1)
        return node

    def _best_split(self, records, parent_node):
        parent_impurity = impurity(parent_node.class_counts, self.criterion)
        n_total = len(records)
        best = None

        for feature in self.feature_names:
            ftype = self.feature_types[feature]

            if ftype == "numeric":
                values = sorted({r[feature] for r in records})
                if len(values) < 2:
                    continue
                # candidate thresholds = midpoints between consecutive unique values
                candidates = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
                for thr in candidates:
                    yes_recs = [r for r in records if r[feature] <= thr]
                    no_recs = [r for r in records if r[feature] > thr]
                    cand = self._score_split(
                        yes_recs, no_recs, n_total, parent_impurity,
                        feature, "numeric", threshold=thr,
                    )
                    if cand and (best is None or cand["gain"] > best["gain"]):
                        best = cand

            else:  # categorical: "is feature == category?"
                categories = sorted({r[feature] for r in records})
                if len(categories) < 2:
                    continue
                for cat in categories:
                    yes_recs = [r for r in records if r[feature] == cat]
                    no_recs = [r for r in records if r[feature] != cat]
                    cand = self._score_split(
                        yes_recs, no_recs, n_total, parent_impurity,
                        feature, "categorical", category=cat,
                    )
                    if cand and (best is None or cand["gain"] > best["gain"]):
                        best = cand

        return best

    def _score_split(self, yes_recs, no_recs, n_total, parent_impurity,
                      feature, feature_type, threshold=None, category=None):
        if len(yes_recs) < self.min_samples_leaf or len(no_recs) < self.min_samples_leaf:
            return None
        yes_counts = counts_of(yes_recs, self.target_key)
        no_counts = counts_of(no_recs, self.target_key)
        weighted = (
            (len(yes_recs) / n_total) * impurity(yes_counts, self.criterion)
            + (len(no_recs) / n_total) * impurity(no_counts, self.criterion)
        )
        gain = parent_impurity - weighted
        return {
            "feature": feature, "feature_type": feature_type,
            "threshold": threshold, "category": category,
            "gain": gain, "weighted_impurity": weighted,
            "yes_records": yes_recs, "no_records": no_recs,
        }

    # ---------- prediction ----------

    def _traverse(self, record, node: Node, path: list):
        if node.is_leaf:
            path.append({"node_id": node.id, "leaf": True, "prediction": node.prediction})
            return node
        if node.feature_type == "numeric":
            went_yes = record[node.feature] <= node.threshold
            question = f"{node.feature} <= {round(node.threshold, 2)}?"
        else:
            went_yes = record[node.feature] == node.category
            question = f"{node.feature} == '{node.category}'?"
        path.append({
            "node_id": node.id, "leaf": False, "question": question,
            "answer": "yes" if went_yes else "no",
        })
        return self._traverse(record, node.yes if went_yes else node.no, path)

    def predict_one(self, record):
        path = []
        leaf = self._traverse(record, self.root, path)
        total = sum(leaf.class_counts.values())
        proba = {c: leaf.class_counts.get(c, 0) / total for c in self.classes_}
        return {
            "prediction": leaf.prediction,
            "probabilities": proba,
            "path": path,
            "leaf_node_id": leaf.id,
        }

    def predict(self, records):
        return [self.predict_one(r)["prediction"] for r in records]

    def evaluate(self, records):
        correct = 0
        y_true, y_pred = [], []
        for r in records:
            pred = self.predict_one(r)["prediction"]
            y_true.append(r[self.target_key])
            y_pred.append(pred)
            if pred == r[self.target_key]:
                correct += 1
        acc = correct / len(records) if records else 0.0
        # confusion matrix
        classes = self.classes_
        matrix = {a: {p: 0 for p in classes} for a in classes}
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
        return {"accuracy": acc, "confusion_matrix": matrix, "n": len(records)}

    # ---------- introspection ----------

    def feature_importances(self):
        total = sum(self._importance_accum.values()) or 1.0
        return {f: round(v / total, 4) for f, v in
                sorted(self._importance_accum.items(), key=lambda kv: -kv[1])}

    def to_json(self):
        def node_to_dict(node: Node):
            d = {
                "id": node.id,
                "depth": node.depth,
                "is_leaf": node.is_leaf,
                "n_samples": node.n_samples,
                "class_counts": node.class_counts,
                "prediction": node.prediction,
                "gini": round(node.gini, 4),
                "entropy": round(node.entropy, 4),
                "accuracy": round(node.accuracy, 4),
            }
            if not node.is_leaf:
                d["feature"] = node.feature
                d["feature_type"] = node.feature_type
                d["threshold"] = round(node.threshold, 3) if node.threshold is not None else None
                d["category"] = node.category
                d["gain"] = round(node.gain, 4)
                d["question"] = (
                    f"{node.feature} <= {round(node.threshold, 2)}?"
                    if node.feature_type == "numeric"
                    else f"{node.feature} == '{node.category}'?"
                )
                d["yes"] = node_to_dict(node.yes)
                d["no"] = node_to_dict(node.no)
            return d

        return node_to_dict(self.root)

    def depth(self):
        def d(node):
            if node.is_leaf:
                return node.depth
            return max(d(node.yes), d(node.no))
        return d(self.root)

    def leaf_count(self):
        def c(node):
            if node.is_leaf:
                return 1
            return c(node.yes) + c(node.no)
        return c(self.root)
