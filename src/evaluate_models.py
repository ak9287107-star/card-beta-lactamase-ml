#!/usr/bin/env python3
"""Evaluate saved grouped-fold models and write reproducible metrics/figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
)

LABEL_ORDER = ["penicillinase_proxy", "esbl_proxy", "carbapenemase_proxy"]
MODEL_NAMES = ("random_forest", "rbf_svm", "xgboost")


def evaluate(model_dir: Path, output_dir: Path, n_splits: int) -> None:
    X = np.load(model_dir / "feature_matrix.npy")
    y = np.load(model_dir / "labels.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    importances: dict[str, np.ndarray] = {}

    for model_name in MODEL_NAMES:
        fold_predictions: list[dict[str, object]] = []
        for fold in range(1, n_splits + 1):
            model_path = model_dir / f"{model_name}_fold_{fold}.joblib"
            test_path = model_dir / f"fold_{fold}_test.npy"
            if not model_path.exists() or not test_path.exists():
                continue
            model = joblib.load(model_path)
            test_idx = np.load(test_path)
            predicted = model.predict(X[test_idx]).astype(int)
            fold_predictions.extend(
                {"row_index": int(i), "fold": fold, "observed": int(obs), "predicted": int(pred)}
                for i, obs, pred in zip(test_idx, y[test_idx], predicted)
            )
            all_rows.append({
                "model": model_name, "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], predicted),
                "accuracy": accuracy_score(y[test_idx], predicted),
            })
            estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if hasattr(estimator, "feature_importances_"):
                importances[model_name] = estimator.feature_importances_
        if fold_predictions:
            pred_frame = pd.DataFrame(fold_predictions)
            pred_frame.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
            cm = confusion_matrix(pred_frame["observed"], pred_frame["predicted"], labels=range(3))
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
            plt.xlabel("Predicted proxy class")
            plt.ylabel("Observed proxy class")
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=200)
            plt.close()
            report = classification_report(
                pred_frame["observed"], pred_frame["predicted"], labels=range(3),
                target_names=LABEL_ORDER, output_dict=True, zero_division=0,
            )
            (output_dir / f"{model_name}_classification_report.json").write_text(json.dumps(report, indent=2) + "\n")

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if not metrics.empty:
        summary = metrics.groupby("model")[["balanced_accuracy", "accuracy"]].agg(["mean", "std"]).reset_index()
        summary.to_csv(output_dir / "model_metrics_summary.csv", index=False)

    feature_names = json.loads((model_dir / "feature_names.json").read_text())
    if importances:
        best_name = max(importances, key=lambda name: float(np.mean(importances[name])))
        importance_frame = pd.DataFrame({"feature": feature_names, "importance": importances[best_name]}).sort_values("importance", ascending=False)
        importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)
        top = importance_frame.head(15).sort_values("importance")
        plt.figure(figsize=(7, 5))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Model split importance")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model_dir, args.output_dir, args.splits)


if __name__ == "__main__":
    main()
