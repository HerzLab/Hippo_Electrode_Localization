"""Purpose: Render subfield-only MRI overlays with the approved QC layouts and colors.

Author: Qirui Zhang
Created: 2026-09-20
Adapted for the standalone library: 2026-09-27.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from hippunfold_surface_qc_utils import LABEL_NAMES, LABEL_COLORS


def subfield_rgba(labels):
    """Color anatomical subfields 1 to 6; background, SRLM, and cyst remain transparent."""
    rgba = np.zeros((*labels.shape, 4))
    for value in range(1, 7):
        rgba[labels == value] = to_rgba(LABEL_COLORS[value], alpha=0.55)
    return rgba


def add_subfield_legend(fig):
    """Use the same named colors as the existing Step 5 anatomy figure."""
    handles = [Patch(facecolor=LABEL_COLORS[i], label=LABEL_NAMES[i]) for i in range(1, 7)]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.935),
               ncol=6, frameon=False, labelcolor='white', fontsize=12)


def render_sagittal_subfields(input_root, subject, output):
    """Render subfield segmentation using the matching approved MRI layout."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.3), facecolor='black')
    for row, hemi in enumerate(['L', 'R']):
        base = input_root / f'{subject}'
        image = nib.as_closest_canonical(nib.load(base / 'anat' / f'{subject}_hemi-{hemi}_space-cropT1w_desc-preproc_T1w.nii.gz'))
        data = image.get_fdata(dtype=np.float32)
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_midthickness.surf.gii')
        vertices = nib.affines.apply_affine(np.linalg.inv(image.affine), mesh.darrays[0].data)
        spacing = np.array(image.header.get_zooms())
        bounds = np.vstack([vertices.min(0), vertices.max(0)])
        lower = np.maximum(0, np.floor(bounds[0] - 4 / spacing).astype(int))
        upper = np.minimum(image.shape, np.ceil(bounds[1] + 4 / spacing).astype(int) + 1)
        roi = data[tuple(slice(a, b) for a, b in zip(lower, upper))]
        vmin, vmax = np.percentile(roi[np.isfinite(roi) & (roi > 0)], [1, 99])
        segmentation = nib.as_closest_canonical(nib.load(base / 'anat' / f'{subject}_hemi-{hemi}_space-cropT1w_label-hipp_desc-subfields_atlas-multihist7_dseg.nii.gz'))
        labels = np.asarray(segmentation.dataobj, dtype=np.uint8)
        for col, fraction in enumerate([0.25, 0.5, 0.75]):
            index = int(round(bounds[0, 0] + fraction * np.diff(bounds[:, 0])[0]))
            ax = axes[row, col]
            ax.imshow(data[index].T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
                      extent=(-0.5*spacing[1], (data.shape[1]-0.5)*spacing[1], -0.5*spacing[2], (data.shape[2]-0.5)*spacing[2]), interpolation='nearest')
            ax.imshow(subfield_rgba(labels[index].T), origin='lower',
                      extent=(-0.5*spacing[1], (data.shape[1]-0.5)*spacing[1], -0.5*spacing[2], (data.shape[2]-0.5)*spacing[2]), interpolation='nearest')
            ax.set_xlim(lower[1]*spacing[1], upper[1]*spacing[1])
            ax.set_ylim(lower[2]*spacing[2], upper[2]*spacing[2])
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'{"Left" if hemi == "L" else "Right"} | Crop slice {index}', color='white', fontsize=13, pad=7)
            x0, x1 = ax.get_xlim()
            z0, z1 = ax.get_ylim()
            ax.plot([x0+2, x0+7], [z0+1.6, z0+1.6], color='white', lw=2)
            ax.text(x0+4.5, z0+2.3, '5 mm', ha='center', color='white', fontsize=9)
            ax.text(0.01, 0.52, 'P', transform=ax.transAxes, color='white')
            ax.text(0.96, 0.52, 'A', transform=ax.transAxes, color='white')
    fig.suptitle(f'{subject} | MRI + hippocampal subfields', color='white', fontsize=21, x=0.035, ha='left', y=0.98)
    add_subfield_legend(fig)
    fig.text(0.035, 0.027, 'Subfields only | Fixed Step 5 colors | 55% opacity | Original crop and slices', color='#cccccc', fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.075, top=0.885, wspace=0.06, hspace=0.16)
    fig.savefig(output / f'{subject}_sagittal_subfields.png', dpi=170, facecolor='black')
    plt.close(fig)




def render_coronal_subfields(input_root, subject, output):
    """Render subfield segmentation using the matching approved MRI layout."""
    base = input_root / f'{subject}'
    image = nib.load(base / 'anat' / f'{subject}_desc-preproc_T1w.nii.gz')
    data = image.get_fdata(dtype=np.float32)
    segmentations, vertex_arrays = [], []
    for hemi in ['L', 'R']:
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_midthickness.surf.gii')
        vertices, faces = mesh.darrays[0].data, mesh.darrays[1].data
        vertex_arrays.append(vertices)
        segmentation = nib.load(base / 'anat' / f'{subject}_hemi-{hemi}_space-cropT1w_label-hipp_desc-subfields_atlas-multihist7_dseg.nii.gz')
        segmentations.append((np.asarray(segmentation.dataobj, dtype=np.uint8), np.linalg.inv(segmentation.affine)))
    vertices = np.concatenate(vertex_arrays)
    lower, upper = vertices.min(0), vertices.max(0)
    x = np.arange(lower[0]-4, upper[0]+4, 0.2)
    z = np.arange(lower[2]-4, upper[2]+4, 0.2)
    xx, zz = np.meshgrid(x, z)
    positions = np.linspace(upper[1]-0.05*(upper[1]-lower[1]), lower[1]+0.05*(upper[1]-lower[1]), 12)
    planes = []
    for y in positions:
        world = np.stack([xx, np.full_like(xx, y), zz], axis=-1)
        voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), world)
        planes.append(map_coordinates(data, voxel.reshape(-1, 3).T, order=1).reshape(xx.shape))
    intensities = np.stack(planes)
    vmin, vmax = np.percentile(intensities[np.isfinite(intensities) & (intensities > 0)], [1, 99])
    fig, axes = plt.subplots(3, 4, figsize=(19, 10), facecolor='black')
    for panel, (ax, y, background) in enumerate(zip(axes.flat, positions, planes)):
        extent = (x[0]-0.1, x[-1]+0.1, z[0]-0.1, z[-1]+0.1)
        ax.imshow(background, cmap='gray', origin='lower', extent=extent, vmin=vmin, vmax=vmax, interpolation='nearest')
        world = np.stack([xx, np.full_like(xx, y), zz], axis=-1)
        for labels, inverse_affine in segmentations:
            voxel = nib.affines.apply_affine(inverse_affine, world)
            sampled = map_coordinates(labels, voxel.reshape(-1, 3).T, order=0,
                                      mode='constant', cval=0, prefilter=False).reshape(xx.shape)
            ax.imshow(subfield_rgba(sampled), origin='lower', extent=extent, interpolation='nearest')
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'{panel+1:02d} | y = {y:.1f} mm', color='white', fontsize=12, pad=6)
        ax.text(0.015, 0.5, 'L', transform=ax.transAxes, color='white', fontsize=11)
        ax.text(0.96, 0.5, 'R', transform=ax.transAxes, color='white', fontsize=11)
        ax.plot([x[0]+2, x[0]+7], [z[0]+1.5]*2, color='white', lw=2)
        ax.text(x[0]+4.5, z[0]+2.3, '5 mm', color='white', ha='center', fontsize=8)
    fig.suptitle(f'{subject} | Coronal MRI + hippocampal subfields', x=0.025, ha='left', y=0.98, color='white', fontsize=23)
    add_subfield_legend(fig)
    fig.text(0.025, 0.026, 'Anterior to posterior, left to right then next row | L on left | Subfields only | Fixed Step 5 colors | 55% opacity', color='#cccccc', fontsize=11)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.88, bottom=0.065, wspace=0.08, hspace=0.18)
    fig.savefig(output / f'{subject}_coronal_subfields.png', dpi=170, facecolor='black')
    plt.close(fig)
