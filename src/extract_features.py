#!/usr/bin/env python3
"""Extract the 26 compact sequence-derived features used in the study."""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = tuple("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = set(AA)
LOGGER = logging.getLogger(__name__)


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = re.sub(r"^>[^\n]*\n", "", text)
    return re.sub(r"[^A-Z]", "", text)


def feature_row(sequence: str) -> dict[str, float]:
    if not sequence:
        raise ValueError("empty sequence")
    analysis = ProteinAnalysis(sequence)
    composition = analysis.get_amino_acids_percent()
    row = {f"aa_{aa}": float(composition.get(aa, 0.0)) for aa in AA}
    row.update(
        molecular_weight=float(analysis.molecular_weight()),
        isoelectric_point=float(analysis.isoelectric_point()),
        aromaticity=float(analysis.aromaticity()),
        instability_index=float(analysis.instability_index()),
        gravy=float(analysis.gravy()),
        charged_catalytic_ratio=float(sum(sequence.count(aa) for aa in "SKED") / len(sequence)),
    )
    return row


def find_sequence_column(frame: pd.DataFrame) -> str:
    for name in ("sequence", "protein_sequence", "aa_sequence", "seq"):
        if name in frame.columns:
            return name
    raise ValueError("Input table must contain sequence, protein_sequence, aa_sequence, or seq")


def extract_features(input_path: Path, output_path: Path, exclusion_path: Path) -> None:
    frame = pd.read_csv(input_path)
    sequence_column = find_sequence_column(frame)
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []

    for index, original in frame[sequence_column].items():
        sequence = normalize_sequence(original)
        invalid = sorted(set(sequence) - VALID_AA)
        reason = None
        if not sequence:
            reason = "empty_sequence"
        elif invalid:
            reason = f"nonstandard_residues:{','.join(invalid)}"
        if reason:
            exclusions.append({"row_index": index, "reason": reason, "sequence": str(original)})
            continue
        try:
            row = frame.loc[index].to_dict()
            row[sequence_column] = sequence
            row.update(feature_row(sequence))
            rows.append(row)
        except Exception as exc:
            exclusions.append({"row_index": index, "reason": f"feature_error:{exc}", "sequence": str(original)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    pd.DataFrame(exclusions).to_csv(exclusion_path, index=False)
    metadata = {
        "amino_acid_order": list(AA),
        "feature_count": 26,
        "feature_names": [f"aa_{aa}" for aa in AA] + [
            "molecular_weight", "isoelectric_point", "aromaticity",
            "instability_index", "gravy", "charged_catalytic_ratio"
        ],
        "charged_catalytic_residues": "S,K,E,D",
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOGGER.info("Wrote %d feature rows and %d exclusions", len(rows), len(exclusions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    extract_features(args.input, args.output, args.exclusions)


if __name__ == "__main__":
    main()
