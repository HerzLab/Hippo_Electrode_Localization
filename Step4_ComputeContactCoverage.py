"""Purpose: Save native contact coverage, intrinsic locations, subfields, and participant summaries.

Author: Qirui Zhang
Created: 2026-09-21
Run only for externally inspected fixed inputs with complete corrected T1 coordinates.
Adapted for the standalone library: 2026-09-27.
"""

# %% Configuration and explicit participant selection
from pathlib import Path
import argparse
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "utils"))
from contact_coverage_io import save_surface_metric, save_contact_table
from utils.pipeline_inputs import read_coordinates, reviewed_subjects, positive_radius

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--external-qc", action="store_true", help="Confirm external QC of all current inputs and use results/localization_qc/QC_fail.csv without Step2 records.")
parser.add_argument("--subjects", nargs="+",
                    help="Exact retained subject IDs with externally inspected coordinates and 2k surfaces.")
parser.add_argument("--radius-mm", type=positive_radius, default=5.0, help="Inclusive contact coverage radius in millimeters (default: 5).")
args = parser.parse_args()
radius_mm = args.radius_mm
result_root = project_root / 'results'
contact_file = project_root / 'data/coordinates.csv'
output_table = result_root / 'localization_hippo.xlsx'
contacts = read_coordinates(project_root)
subjects = reviewed_subjects(project_root, contacts, args.subjects, external_qc=args.external_qc)
output_table.parent.mkdir(parents=True, exist_ok=True)

# %% Fixed source rows and output schema
contacts["source_row"] = np.arange(2, len(contacts) + 2)  # One-based CSV row, including the header.
contacts["contact_id"] = contacts["source_row"].map(lambda row: f"row{row:06d}")
output_columns = ["subject", "lead", "contact",
                  "x_mm", "y_mm", "z_mm",
                  "hemisphere", "min_vertex_distance_mm", "n_covered_vertices",
                  "radius_mm", "nearest_vertex_ap", "nearest_vertex_pd",
                  "coverage_ap_area_mean", "coverage_pd_area_mean", "nearest_vertex_subfield",
                  "coverage_subiculum_area_pct", "coverage_ca1_area_pct", "coverage_ca2_area_pct",
                  "coverage_ca3_area_pct", "coverage_ca4_area_pct"]
subfield_names = {0: "", 1: "Subiculum", 2: "CA1", 3: "CA2", 4: "CA3", 5: "CA4"}
volume_regions = {"Sub": "subiculum", "CA1": "ca1", "CA2": "ca2", "CA3": "ca3",
                  "CA4": "ca4", "DG": "dg", "SRLM": "srlm", "Cyst": "cyst"}
subject_fields = ["hippocampus_tissue_volume_mm3", "hippocampus_gm_volume_mm3",
                  *[f"{region}_volume_mm3" for region in volume_regions.values()],
                  "n_contacts", "n_covered_vertices", "coverage_vertex_pct"]
subject_columns = ["subject", *[f"{side}_{field}" for side in ["left", "right"] for field in subject_fields]]
output_rows = []
participant_rows = []

# %% Native Euclidean neighborhoods, separately for both hemispheres
for subject in subjects:
    subject_contacts = contacts.loc[contacts["subject"].eq(subject)]
    centers = subject_contacts[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    contact_dir = result_root / "contact_overlap" / f"{subject}"
    contact_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    participant_row = {"subject": subject}
    anatomy_root = result_root / "hippunfold" / f"{subject}"
    volumes = pd.read_csv(anatomy_root / "anat" / f"{subject}_space-cropT1w_desc-subfields_atlas-multihist7_volumes.tsv",
                          sep="\t", float_precision="round_trip").set_index(["subject", "hemi"])
    for hemisphere in ["L", "R"]:
        prefix = f"{subject}_hemi-{hemisphere}_space-T1w_den-2k_label-hipp"
        surface_file = result_root / "hippunfold" / f"{subject}" / "surf" / f"{prefix}_midthickness.surf.gii"
        mesh = nib.load(surface_file)
        vertices = mesh.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data.astype(np.float64)
        faces = mesh.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
        triangle_areas = 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                                      vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
        vertex_areas = np.zeros(len(vertices), dtype=np.float64)
        np.add.at(vertex_areas, faces.ravel(), np.repeat(triangle_areas / 3.0, 3))  # Barycentric native surface areas in mm2.
        metric_root = anatomy_root / "metric"
        ap = nib.load(metric_root / f"{subject}_dir-AP_hemi-{hemisphere}_den-2k_label-hipp_desc-laplace_coords.shape.gii").darrays[0].data.astype(np.float64)
        pd_coords = nib.load(metric_root / f"{subject}_dir-PD_hemi-{hemisphere}_den-2k_label-hipp_desc-laplace_coords.shape.gii").darrays[0].data.astype(np.float64)
        subfields = nib.load(metric_root / f"{subject}_hemi-{hemisphere}_den-2k_label-hipp_atlas-multihist7_subfields.label.gii").darrays[0].data
        distance_mm = cdist(vertices, centers, metric="euclidean")  # Rows are original vertices; columns are source contacts.
        membership = distance_mm <= radius_mm
        retained = np.flatnonzero(membership.any(axis=0))
        metadata = {"Subject": subject, "Hemisphere": hemisphere, "Structure": "hipp",
                    "SurfaceDensity": "2k", "CoordinateSpace": "T1w", "RadiusMM": str(radius_mm),
                    "SourceSurface": surface_file.relative_to(project_root).as_posix(),
                    "SourceContacts": contact_file.relative_to(project_root).as_posix(),
                    "CoordinateVariant": "native_T1_RAS_mm", "VertexOrder": "Original source surface order",
                    "CoverageRule": f"Euclidean distance <= {radius_mm} mm", "Author": "Qirui Zhang"}
        for contact_index in retained:
            row = subject_contacts.iloc[contact_index].to_dict()
            stem = f"{prefix}_desc-contact{row['contact']}"
            distance_file = contact_dir / f"{stem}_distance.shape.gii"
            coverage_file = contact_dir / f"{stem}_coverage.shape.gii"
            mask = membership[:, contact_index]
            nearest_vertex = int(np.argmin(distance_mm[:, contact_index]))  # Exact ties retain the first original vertex.
            covered_area = vertex_areas[mask].sum()
            values = np.where(mask, distance_mm[:, contact_index], 0.0).astype(np.float32)
            contact_metadata = dict(metadata, ContactID=row["contact_id"], Contact=str(row["contact"]),
                                    Lead=str(row["lead"]), SourceRow=str(row["source_row"]))
            save_surface_metric(distance_file, values, dict(contact_metadata, Units="mm", OutsideCoverage="0"))
            save_surface_metric(coverage_file, mask.astype(np.uint8), dict(contact_metadata, Units="binary"))
            row.update(hemisphere=hemisphere, min_vertex_distance_mm=float(distance_mm[:, contact_index].min()),
                       n_covered_vertices=int(mask.sum()), radius_mm=radius_mm,
                       nearest_vertex_ap=float(ap[nearest_vertex]), nearest_vertex_pd=float(pd_coords[nearest_vertex]),
                       coverage_ap_area_mean=float(np.sum(vertex_areas[mask] * ap[mask]) / covered_area),
                       coverage_pd_area_mean=float(np.sum(vertex_areas[mask] * pd_coords[mask]) / covered_area),
                       nearest_vertex_subfield=subfield_names[int(subfields[nearest_vertex])],
                       reference_image=f"data/native_T1/{subject}_T1w.nii.gz", coordinate_variant="native_T1_RAS_mm",
                       surface_file=metadata["SourceSurface"], distance_file=distance_file.relative_to(project_root).as_posix(),
                       coverage_file=coverage_file.relative_to(project_root).as_posix())
            for label in [1, 2, 3, 4, 5]:
                row[f"coverage_{subfield_names[label].lower()}_area_pct"] = float(100.0 * vertex_areas[mask & (subfields == label)].sum() / covered_area)  # Full covered area denominator; no unassigned output column.
            subject_rows.append(row)
        overlap = membership.any(axis=1).astype(np.uint8)  # Includes true zero-distance vertices; empty coverage remains zero.
        overlap_file = contact_dir / f"{prefix}_desc-contactUnion_coverage.shape.gii"
        save_surface_metric(overlap_file, overlap, dict(metadata, Units="binary", Aggregation="Within-subject union",
                                                      RetainedContacts=str(len(retained))))
        side = {"L": "left", "R": "right"}[hemisphere]
        hemisphere_volumes = volumes.loc[(f"{subject}", hemisphere)]
        gm_volume = float(hemisphere_volumes[["Sub", "CA1", "CA2", "CA3", "CA4", "DG"]].sum())
        participant_row[f"{side}_hippocampus_gm_volume_mm3"] = gm_volume
        participant_row[f"{side}_hippocampus_tissue_volume_mm3"] = gm_volume + float(hemisphere_volumes["SRLM"])
        for source_name, region in volume_regions.items():
            participant_row[f"{side}_{region}_volume_mm3"] = float(hemisphere_volumes[source_name])
        participant_row[f"{side}_n_contacts"] = int(len(retained))
        participant_row[f"{side}_n_covered_vertices"] = int(overlap.sum())
        participant_row[f"{side}_coverage_vertex_pct"] = float(100.0 * overlap.sum() / len(vertices))
        print(f"{subject} {hemisphere}: {len(retained)} contacts; {int(overlap.sum())} covered vertices", flush=True)
    output_rows.extend(sorted(subject_rows, key=lambda row: (row["source_row"], row["hemisphere"])))
    participant_rows.append(participant_row)

# %% One fixed derivative table for the explicitly selected run
save_contact_table(output_table, output_rows, output_columns, participant_rows, subject_columns)
print(f"Saved {len(output_rows)} contact-hemisphere rows to {output_table.relative_to(project_root)}", flush=True)
