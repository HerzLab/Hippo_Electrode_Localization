"""Purpose: Generate individual 2k surface and MRI contact-overlay figures for visual QC.

Author: Qirui Zhang
Created: 2026-09-20
Adapted from the approved individual_surface_preview pilot.
Adapted for the standalone library: 2026-09-27.
"""

# %% Configuration
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from utils.pipeline_inputs import read_coordinates, reviewed_subjects, positive_radius

project_root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--external-qc", action="store_true", help="Confirm external QC of all current inputs and use results/localization_qc/QC_fail.csv without Step2 records.")
parser.add_argument("--subjects", nargs="+",
                    help="Exact retained subject IDs with existing 2k HippUnfold outputs.")
parser.add_argument("--radius-mm", type=positive_radius, help="Require this radius in saved Step4 maps; default: read the saved radius.")
args = parser.parse_args()
sys.path.insert(0, str(project_root / "utils"))
from hippunfold_ribbon_qc_utils import render_sagittal_ribbon, render_coronal_ribbon
from hippunfold_subfield_qc_utils import render_sagittal_subfields, render_coronal_subfields
from hippunfold_mri_qc_utils import render_sagittal_overlay, render_coronal_overlay
from hippunfold_surface_qc_utils import load_surfaces, load_metric, render_row, assemble_figure, LABEL_CMAP
from hippunfold_contact_qc_utils import render_contact_map

contacts = read_coordinates(project_root)
subjects = reviewed_subjects(project_root, contacts, args.subjects, external_qc=args.external_qc)
input_root = project_root / "results" / "hippunfold"
output_root = project_root / "results" / "visual_qc"
contact_root = project_root / "results" / "contact_overlap"
render_dir = project_root / "tmp" / "hippomaps_rendering" / "den-2k" / "renders"
render_dir.mkdir(parents=True, exist_ok=True)
morphology = [("thickness", "Thickness (mm)", "viridis", (0, 4)),
              ("gyrification", "Gyrification", "magma", (0, 5)),
              ("curvature", "Curvature (native output scale)", "coolwarm", (-1.6, 1.6))]
coordinates = [("AP", "Anterior-posterior coordinate", "viridis", (0, 1)),
               ("PD", "Proximal-distal coordinate", "plasma", (0, 1))]

# %% Fixed corrected T1 contact input
hippo_contacts = pd.read_excel(project_root / "results" / "localization_hippo.xlsx", sheet_name="Contacts", keep_default_na=False, dtype={"subject": str, "lead": str, "contact": str})
coverage_subjects = pd.read_excel(project_root / "results" / "localization_hippo.xlsx", sheet_name="subjects", dtype={"subject": str})
if not set(subjects).issubset(set(coverage_subjects.subject)):
    raise ValueError("Run Step 4 for every selected subject before visualization.")
saved_radii = set()
for subject in subjects:
    for hemi in ["L", "R"]:
        path = contact_root / subject / f"{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_desc-contactUnion_coverage.shape.gii"
        saved_radii.add(positive_radius(nib.load(path).meta["RadiusMM"]))
if len(saved_radii) != 1:
    raise ValueError("Selected Step4 maps contain mixed coverage radii. Recompute the selection together.")
radius_mm = saved_radii.pop()
if args.radius_mm is not None and args.radius_mm != radius_mm:
    raise ValueError("Requested radius does not match saved Step4 results. Recompute coverage before plotting.")
selected_rows = hippo_contacts.loc[hippo_contacts.subject.isin(subjects)]
if not selected_rows.radius_mm.eq(radius_mm).all():
    raise ValueError("The coverage workbook and surface maps have different radii.")

# %% Individual anatomy, morphology, intrinsic coordinates, and MRI overlays
for subject in subjects:
    print(f"Rendering {subject}", flush=True)
    output_dir = output_root / f"{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_output_dir = output_dir / "contact_distance"
    contact_output_dir.mkdir(parents=True, exist_ok=True)
    hippocampus = load_surfaces(input_root, subject, "hipp")
    dentate = load_surfaces(input_root, subject, "dentate")
    labels = load_metric(input_root, subject, "subfields")
    anatomy_rows = []
    for structure, surfaces, values, label in [
        ("hipp", hippocampus, labels, "Hippocampus | atlas-informed subfields"),
        ("dentate", dentate, {h: np.full(dentate[h + "f"].n_points, 6, dtype=np.int32) for h in ["L", "R"]}, "Dentate gyrus | displayed separately")]:
        path = render_dir / f"{subject}_{structure}_subfields.png"
        render_row(surfaces, values, path, LABEL_CMAP, (-0.5, 6.5))
        anatomy_rows.append((path, label, LABEL_CMAP, (-0.5, 6.5)))
    assemble_figure(subject, anatomy_rows, output_dir / f"{subject}_anatomy.png", "Individual anatomy",
                    "HippUnfold 2k | Individual midthickness surfaces | Panels fitted independently; sizes are not directly comparable.", categorical=True)
    for family, configuration, title, footer in [
        ("morphology", morphology, "Surface morphology", "Hippocampal surface | Identical scales across participants and hemispheres | Original values; no smoothing."),
        ("coordinates", coordinates, "Intrinsic coordinates", "Hippocampal surface | Normalized coordinates, not millimeters | Native v2 orientation; no AP rescaling.")]:
        rows = []
        for feature, label, cmap, limits in configuration:
            values = load_metric(input_root, subject, feature)
            path = render_dir / f"{subject}_{feature}.png"
            render_row(hippocampus, values, path, cmap, limits)
            rows.append((path, label, cmap, limits))
        assemble_figure(subject, rows, output_dir / f"{subject}_{family}.png", title, footer)
    subject_contacts = contacts.loc[contacts["subject"].eq(subject)]
    centers = subject_contacts[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    render_sagittal_overlay(input_root, subject, centers, output_dir, radius_mm=radius_mm)
    render_coronal_overlay(input_root, subject, centers, output_dir, radius_mm=radius_mm)
    render_sagittal_subfields(input_root, subject, output_dir)
    render_coronal_subfields(input_root, subject, output_dir)
    render_sagittal_ribbon(input_root, subject, output_dir)
    render_coronal_ribbon(input_root, subject, output_dir)

    # Display saved Step 4 metrics on the matching original vertex topology.
    union_values = {}
    for hemi in ["L", "R"]:
        prefix = f"{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp"
        path = contact_root / f"{subject}" / f"{prefix}_desc-contactUnion_coverage.shape.gii"
        union_values[hemi] = nib.load(path).darrays[0].data.copy()
    render_contact_map(hippocampus, union_values, render_dir / f"{subject}_contactUnion_row.png",
                       output_dir / f"{subject}_contactUnion_coverage.png", subject,
                       "Contact coverage union", binary=True, radius_mm=radius_mm)
    selected_contacts = hippo_contacts.loc[hippo_contacts["subject"].eq(subject)]
    for contact, rows in selected_contacts.groupby("contact", sort=False):
        distance_values = {hemi: np.full(hippocampus[hemi + "f"].n_points, np.nan) for hemi in ["L", "R"]}
        for hemi in rows["hemisphere"]:
            prefix = f"{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_desc-contact{contact}"
            distance_image = nib.load(contact_root / f"{subject}" / f"{prefix}_distance.shape.gii")
            coverage_image = nib.load(contact_root / f"{subject}" / f"{prefix}_coverage.shape.gii")
            if any(float(image.meta["RadiusMM"]) != radius_mm for image in [distance_image, coverage_image]):
                raise ValueError(f"{subject} {contact}: stale contact maps have a different radius.")
            distance = distance_image.darrays[0].data
            coverage = coverage_image.darrays[0].data
            distance_values[hemi] = np.where(coverage == 1, distance, np.nan)  # Mask by coverage, preserving real zero distances.
        render_contact_map(hippocampus, distance_values, render_dir / f"{subject}_contact-{contact}_row.png",
                           contact_output_dir / f"{subject}_contact-{contact}_distance.png", subject,
                           f"Contact {contact} | Euclidean distance", radius_mm=radius_mm)
    print(f"Completed {subject}", flush=True)
