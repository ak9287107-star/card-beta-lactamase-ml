#!/usr/bin/env python3
"""Train compact-feature classifiers with family-grouped cross-validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # allow RF/SVM use when xgboost is unavailable
    XGBClassifier = None

FEATURES = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"] + [
    "molecular_weight", "isoelectric_point", "aromaticity", "instability_index",
    "gravy", "charged_catalytic_ratio",
]
LABEL_ORDER = ["penicillinase_proxy", "esbl_proxy", "carbapenemase_proxy"]


def build_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "rbf_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
        ]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            eval_metric="mlogloss", random_state=seed, n_jobs=-1,
        )
    return models


def train(input_path: Path, output_dir: Path, n_splits: int, seed: int) -> None:
    frame = pd.read_csv(input_path)
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    for required in ("spectrum_proxy_label", "group_id"):
        if required not in frame.columns:
            raise ValueError(f"Input must contain {required!r}")

    frame = frame.dropna(subset=FEATURES + ["spectrum_proxy_label", "group_id"]).copy()
    label_to_int = {label: i for i, label in enumerate(LABEL_ORDER)}
    unknown = sorted(set(frame["spectrum_proxy_label"]) - set(label_to_int))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    X = frame[FEATURES].astype(float)
    y = frame["spectrum_proxy_label"].map(label_to_int).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "feature_matrix.npy", X.to_numpy())
    np.save(output_dir / "labels.npy", y)
    frame[["spectrum_proxy_label", "group_id"]].to_csv(output_dir / "model_index.csv", index=False)
    (output_dir / "feature_names.json").write_text(json.dumps(FEATURES, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_to_int, indent=2) + "\n")

    fold_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        np.save(output_dir / f"fold_{fold}_train.npy", train_idx)
        np.save(output_dir / f"fold_{fold}_test.npy", test_idx)
        for name, model in build_models(seed + fold).items():
            model.fit(X.iloc[train_idx], y[train_idx])
            joblib.dump(model, output_dir / f"{name}_fold_{fold}.joblib")
            fold_rows.append({"fold": fold, "model": name, "train_size": len(train_idx), "test_size": len(test_idx)})

    pd.DataFrame(fold_rows).to_csv(output_dir / "training_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.input, args.output_dir, args.splits, args.seed)


if __name__ == "__main__":
    main()
