"""Purpose: Read the fixed native T1 contact contract and completed human review.

Author: Qirui Zhang
Created: 2026-09-27
"""

from pathlib import Path
import argparse
import hashlib
import re
import numpy as np
import pandas as pd


COORDINATE_COLUMNS = ["subject", "lead", "contact", "x_mm", "y_mm", "z_mm"]


def positive_radius(value):
    """Accept a finite positive physical radius in millimeters."""
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0:
        raise argparse.ArgumentTypeError("Radius must be finite and greater than zero.")
    return radius


def read_coordinates(root):
    """Preserve identifiers, row order, and missing coordinates for localization QC."""
    table = pd.read_csv(Path(root) / "data/coordinates.csv", dtype=str, keep_default_na=False)
    if list(table.columns) != COORDINATE_COLUMNS or table.empty:
        raise ValueError("coordinates.csv must contain the six documented columns and at least one contact.")
    for column in ["subject", "lead", "contact"]:
        if not table[column].map(lambda value: bool(re.fullmatch(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", value))).all():
            raise ValueError(f"{column} must contain nonempty filename-safe identifiers.")
    if not table.subject.str.fullmatch(r"sub-[A-Za-z0-9]+").all():
        raise ValueError("Subject identifiers must use sub- followed by letters or digits.")
    if table.duplicated(["subject", "contact"]).any():
        raise ValueError("Contact names must be unique within each subject because they identify surface files.")
    for column in ["x_mm", "y_mm", "z_mm"]:
        table[column] = pd.to_numeric(table[column].replace("", np.nan), errors="raise")
    return table


def select_subjects(contacts, requested):
    """Select explicitly requested participants in original coordinate-table order."""
    ordered = contacts.subject.drop_duplicates().tolist()
    if requested is None:
        return ordered
    unknown = set(requested) - set(ordered)
    if unknown:
        raise ValueError(f"Unknown subjects: {sorted(unknown)}")
    return [subject for subject in ordered if subject in requested]


def reviewed_subjects(root, contacts, requested, external_qc=False):
    """Require final review and complete coordinates before downstream processing."""
    qc = Path(root) / "results/localization_qc"
    excluded = pd.read_csv(qc / "QC_fail.csv", dtype=str, keep_default_na=False)
    current = contacts.subject.drop_duplicates().tolist()
    if list(excluded.columns) != ["subject", "reason"]:
        raise ValueError("QC_fail.csv must contain exactly subject,reason columns.")
    if excluded.subject.duplicated().any() or not set(excluded.subject).issubset(current):
        raise ValueError("QC_fail.csv contains duplicate or unknown subject identifiers.")
    if excluded.reason.str.strip().eq("").any():
        raise ValueError("Every excluded participant needs a nonempty reason in QC_fail.csv.")
    if external_qc:
        expected = set(excluded.subject)  # The user explicitly confirms external review of the current inputs.
    else:
        summary = pd.read_csv(qc / "subject_qc.csv", dtype=str, keep_default_na=False)
        manual = pd.read_csv(qc / "manual_qc.csv", dtype=str, keep_default_na=False)
        if summary.subject.tolist() != current:
            raise ValueError("Localization QC does not match coordinates.csv. Repeat Steps 1 and 2.")
        coordinate_hash = hashlib.sha256((Path(root) / "data/coordinates.csv").read_bytes()).hexdigest()
        if not summary.coordinates_sha256.eq(coordinate_hash).all():
            raise ValueError("Coordinates changed after localization QC. Repeat Steps 1 and 2.")
        for row in summary.itertuples():
            image = Path(root) / "data/native_T1" / f"{row.subject}_T1w.nii.gz"
            digest = hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else ""
            if digest != row.t1_sha256:
                raise ValueError(f"{row.subject}: native T1 changed after localization QC.")
        reviewable = summary.loc[summary.image_status.eq("Images generated; visual review pending"), "subject"].tolist()
        if manual.subject.tolist() != reviewable or manual.decision.eq("").any():
            raise ValueError("Complete the localization review in Step 2 before proceeding.")
        failures = {"T1_QC_fail", "Hippo_Surgery", "Other_QC_Fail"}
        if not manual.decision.isin(failures | {"QC Pass", "Other_Surgery_lesion", "T1_Enhanced"}).all():
            raise ValueError("Unknown localization review decision.")
        expected = set(manual.loc[manual.decision.isin(failures), "subject"])
        expected.update(summary.loc[summary.image_status.eq("Missing reference image") | summary.n_valid_coordinates.eq("0"), "subject"])
        if set(excluded.subject) != expected:
            raise ValueError("QC_fail.csv does not match the completed review.")
    selected = select_subjects(contacts, requested)
    if requested is not None and expected.intersection(selected):
        raise ValueError("An explicitly requested participant was excluded by localization QC.")
    selected = [subject for subject in selected if subject not in expected]
    if not selected:
        raise ValueError("No participants remain after localization QC.")
    for subject in selected:
        centers = contacts.loc[contacts.subject.eq(subject), ["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
        if not np.isfinite(centers).all():
            raise ValueError(f"{subject}: complete finite coordinates are required; no contacts are silently skipped.")
        if not (Path(root) / "data/native_T1" / f"{subject}_T1w.nii.gz").is_file():
            raise FileNotFoundError(f"Missing native T1 image for {subject}.")
    return selected
