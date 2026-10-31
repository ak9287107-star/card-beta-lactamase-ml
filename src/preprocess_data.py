#!/usr/bin/env python3
"""Prepare CARD-derived rows for supervised proxy-label modeling.

The proxy labels are exploratory CARD drug-class relationships, not experimental
hydrolysis measurements. The script preserves unknown and ambiguous rows for audit.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABELS = ("penicillinase_proxy", "esbl_proxy", "carbapenemase_proxy")


def first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def text_columns(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        c for c in frame.columns
        if any(token in c.lower() for token in ("drug", "class", "aro", "name", "mechanism", "description"))
    ]
    if not candidates:
        candidates = list(frame.columns)
    return frame[candidates].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()


def ambler_class(text: str) -> str | None:
    match = re.search(r"\b(?:ambler\s*)?class\s*([abcd])\b", text, re.I)
    return match.group(1).upper() if match else None


def proxy_label(text: str) -> str | None:
    # Apply the most specific resistance-spectrum terms first.
    if any(term in text for term in ("carbapenemase", "carbapenem resistance", "carbapenem")):
        return "carbapenemase_proxy"
    if any(term in text for term in ("extended-spectrum beta-lactamase", "extended spectrum beta-lactamase", "esbl", "cephalosporin")):
        return "esbl_proxy"
    if any(term in text for term in ("penicillinase", "penicillin resistance", "penicillin")):
        return "penicillinase_proxy"
    return None


def preprocess(input_path: Path, output_path: Path, audit_path: Path) -> None:
    frame = pd.read_csv(input_path)
    combined = text_columns(frame)
    frame["ambler_class"] = combined.map(ambler_class)
    frame["spectrum_proxy_label"] = combined.map(proxy_label)

    group_col = first_column(frame, ("amr_gene_family", "gene_family", "family", "aro_family", "card_short_name"))
    if group_col:
        frame["group_id"] = frame[group_col].fillna("unknown_group").astype(str)
    else:
        frame["group_id"] = frame.index.map(lambda i: f"row_{i}")

    eligible = frame["ambler_class"].isin(list("ABCD")) & frame["spectrum_proxy_label"].isin(LABELS)
    modeling = frame.loc[eligible].copy()
    audit = frame.loc[~eligible].copy()
    audit["exclusion_reason"] = "missing_or_ambiguous_ambler_class_or_proxy_label"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)
    summary = {
        "input_rows": int(len(frame)),
        "modeling_rows": int(len(modeling)),
        "excluded_rows": int(len(audit)),
        "label_counts": modeling["spectrum_proxy_label"].value_counts().to_dict(),
        "ambler_counts": modeling["ambler_class"].value_counts().to_dict(),
        "group_count": int(modeling["group_id"].nunique()),
        "proxy_definition": "CARD drug-class text rule; not experimentally verified hydrolysis",
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    preprocess(args.input, args.output, args.audit)


if __name__ == "__main__":
    main()
