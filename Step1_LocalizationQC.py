"""Purpose: Draw corrected T1 contact spheres and single overlay coronal QC mosaics.

Author: Qirui Zhang
Created: 2026-09-18
Adapted for the standalone library: 2026-09-27.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import sys
import nibabel as nib
import numpy as np
from utils.pipeline_inputs import read_coordinates, select_subjects, positive_radius

# %% Fixed inputs and display settings
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'utils'))
from localization_qc_utils import sphere_indices, save_native_image, render_coronal_mosaics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subjects', nargs='+', help='Explicit subject IDs for a bounded run; default: all fixed inputs.')
parser.add_argument('--radius-mm', type=positive_radius, default=5.0, help='Contact sphere radius in millimeters (default: 5).')
args = parser.parse_args()
radius_mm = args.radius_mm
radius_label = str(radius_mm).removesuffix(".0")
contacts = read_coordinates(root)
subjects = select_subjects(contacts, args.subjects)
references = [{'subject': subject, 'reference_image': f'data/native_T1/{subject}_T1w.nii.gz'} for subject in subjects]
output_root = root / 'results/localization_qc'
if args.subjects:
    output_root = root / 'tmp/localization_qc_subset'
if (output_root / 'manual_qc.csv').exists():
    raise RuntimeError('Existing human review must be archived before regenerating its input images.')
output_root.mkdir(parents=True, exist_ok=True)
sphere_dir = output_root / f"contact_spheres_r{radius_label}mm"
composite_dir = output_root / f"T1_contact_spheres_r{radius_label}mm"
mosaic_dir = output_root / "coronal_qc"
for folder in [sphere_dir, composite_dir, mosaic_dir]:
    folder.mkdir(exist_ok=True)
coordinate_fields = ['x_mm', 'y_mm', 'z_mm']

# %% Preserve coordinate row order and report missing values explicitly
contacts_by_subject = {subject: contacts.loc[contacts.subject.eq(subject)].to_dict('records') for subject in subjects}
contact_fields = ['subject', 'lead', 'contact',
                  'x_t1_mm', 'y_t1_mm', 'z_t1_mm', 'coordinate_valid', 'reference_image',
                  'i_vox', 'j_vox', 'k_vox', 'center_inside_image', 'sphere_clipped',
                  'sphere_voxels_full', 'sphere_voxels_written', 'render_status', 'radius_mm']
summary_fields = ['subject', 'reference_image', 'n_contacts', 'n_valid_coordinates',
                  'n_missing_coordinates', 'n_centers_outside', 'n_spheres_clipped',
                  'n_spheres_without_voxels', 'n_contacts_rendered', 'sphere_value_p99',
                  'window_low', 'window_high', 'n_slices', 'n_pages', 'image_status',
                  'coordinates_sha256', 't1_sha256', 'radius_mm']
coordinate_hash = hashlib.sha256((root / 'data/coordinates.csv').read_bytes()).hexdigest()

# %% Evaluate scientific localization QC and write each available native image
report_suffix = ''
with (output_root / f'contact_qc{report_suffix}.csv').open('w', newline='', encoding='utf-8') as contact_stream, \
     (output_root / f'subject_qc{report_suffix}.csv').open('w', newline='', encoding='utf-8') as summary_stream:
    contact_writer = csv.DictWriter(contact_stream, fieldnames=contact_fields)
    summary_writer = csv.DictWriter(summary_stream, fieldnames=summary_fields)
    contact_writer.writeheader()
    summary_writer.writeheader()
    for number, item in enumerate(references, 1):
        subject = item['subject']
        contacts = contacts_by_subject[subject]
        centers = np.array([[np.nan if row[field] is None else row[field]
                             for field in coordinate_fields] for row in contacts], dtype=float)
        valid = np.isfinite(centers).all(axis=1)
        summary = dict.fromkeys(summary_fields, '')
        summary.update(subject=subject, reference_image=item['reference_image'], n_contacts=len(contacts),
                       n_valid_coordinates=int(valid.sum()), n_missing_coordinates=int((~valid).sum()),
                       coordinates_sha256=coordinate_hash, radius_mm=radius_mm)
        details = []
        for row, center, finite in zip(contacts, centers, valid):
            detail = dict.fromkeys(contact_fields, '')
            detail.update(subject=subject,
                          lead=row['lead'], contact=row['contact'], coordinate_valid=bool(finite),
                          reference_image=item['reference_image'], radius_mm=radius_mm)
            detail.update({field: float(value) if np.isfinite(value) else '' for field, value in
                           zip(['x_t1_mm', 'y_t1_mm', 'z_t1_mm'], center)})
            details.append(detail)
        if not (root / item['reference_image']).is_file():
            # Report missing native T1 references without substituting another image.
            summary['image_status'] = 'Missing reference image'
            for detail in details:
                detail['render_status'] = 'Missing reference image'
        else:
            summary['t1_sha256'] = hashlib.sha256((root / item['reference_image']).read_bytes()).hexdigest()
            reference = nib.load(root / item['reference_image'])
            data = reference.get_fdata(dtype=np.float32)
            shape = np.array(data.shape)
            affine = reference.get_sform()
            if int(reference.header['sform_code']) == 0 or reference.header.get_xyzt_units()[0] != 'mm':
                raise ValueError(f'{subject}: a valid sform in millimeters is required.')
            mask = np.zeros(data.shape, dtype=np.uint8)
            for detail, center, finite in zip(details, centers, valid):
                if not finite:
                    detail['render_status'] = 'Missing or nonfinite corrected coordinate'
                    continue
                indices, voxel, inside, clipped, full_count = sphere_indices(center, affine, shape, radius_mm)
                mask[tuple(indices.T)] = 1
                detail.update(i_vox=voxel[0], j_vox=voxel[1], k_vox=voxel[2],
                              center_inside_image=inside, sphere_clipped=clipped,
                              sphere_voxels_full=full_count, sphere_voxels_written=len(indices),
                              render_status='Rendered' if len(indices) else 'No sphere voxel in image')
            summary.update(n_centers_outside=sum(d['center_inside_image'] is False for d in details),
                           n_spheres_clipped=sum(d['sphere_clipped'] is True for d in details),
                           n_spheres_without_voxels=sum(d['sphere_voxels_written'] == 0 for d in details),
                           n_contacts_rendered=sum(d['render_status'] == 'Rendered' for d in details))
            if not valid.any():
                summary['image_status'] = 'No valid corrected coordinates'
            else:
                intensities = data[np.isfinite(data) & (data != 0)]
                low, high = np.percentile(intensities, [1, 99])
                save_native_image(mask, reference, sphere_dir / f'{subject}.nii.gz', np.uint8, (0, 1))
                composite = data.copy()
                composite[mask > 0] = high
                for center in centers[valid]:
                    inner_indices, *_ = sphere_indices(center, affine, shape, min(2.0, 0.4 * radius_mm))
                    composite[tuple(inner_indices.T)] = 0
                save_native_image(composite, reference, composite_dir / f'{subject}.nii.gz', np.float32, (low, high))
                n_slices, n_pages = render_coronal_mosaics(data, affine, centers[valid], mosaic_dir / f'{subject}.png',
                                                          subject, (low, high), radius_mm)
                summary.update(sphere_value_p99=high, window_low=low, window_high=high,
                               n_slices=n_slices, n_pages=n_pages,
                               image_status='Images generated; visual review pending')
        contact_writer.writerows(details)
        summary_writer.writerow(summary)
        contact_stream.flush()
        summary_stream.flush()
        print(f'{number}/{len(references)} {subject}: {summary["image_status"]}', flush=True)
