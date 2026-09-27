"""Purpose: Overlay hippocampal inner, midthickness, and outer intersections on MRI.

Author: Qirui Zhang
Created: 2026-09-21
Layout adapted from the approved subfield-only MRI QC views.
Adapted for the standalone library: 2026-09-27.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SURFACE_COLORS = {'inner': '#ffff00', 'midthickness': '#ff9900', 'outer': '#ff0000'}


def load_ribbon_surfaces(base, subject, hemi, transform):
    """Read original 2k meshes in the coordinate frame of the displayed slice planes."""
    surfaces = {}
    for name in SURFACE_COLORS:
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_{name}.surf.gii')
        vertices = nib.affines.apply_affine(transform, mesh.darrays[0].data)
        faces = mesh.darrays[1].data
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel(), deep=True))
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(cells)
        surfaces[name] = surface
    return surfaces


def draw_ribbon_intersections(ax, surfaces, axis, position, display_axes, scale):
    """Draw plane intersections without projecting, smoothing, or closing open contours."""
    normal = np.eye(3)[axis]
    for name, surface in surfaces.items():
        plane = vtk.vtkPlane()
        plane.SetNormal(*normal)
        plane.SetOrigin(*(normal * position))
        cutter = vtk.vtkCutter()
        cutter.SetInputData(surface)
        cutter.SetCutFunction(plane)
        cutter.Update()
        section = cutter.GetOutput()
        if section.GetNumberOfPoints() == 0:
            continue  # A surface may end before a displayed slice plane.
        points = vtk_to_numpy(section.GetPoints().GetData())
        segments = []
        for i in range(section.GetNumberOfCells()):
            cell = section.GetCell(i)
            segment = points[[cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]]
            segments.append(segment[:, display_axes] * scale)
        ax.add_collection(LineCollection(segments, colors=SURFACE_COLORS[name], linewidths=1.05))


def add_ribbon_legend(fig):
    handles = [Line2D([0], [0], color=color, lw=1.5, label=name.capitalize()) for name, color in SURFACE_COLORS.items()]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.935),
               ncol=3, frameon=False, labelcolor='white', fontsize=12)


def render_sagittal_ribbon(input_root, subject, output):
    """Render inner, middle, and outer intersections in the matching approved MRI layout."""
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
        surfaces = load_ribbon_surfaces(base, subject, hemi, np.linalg.inv(image.affine))
        for col, fraction in enumerate([0.25, 0.5, 0.75]):
            index = int(round(bounds[0, 0] + fraction * np.diff(bounds[:, 0])[0]))
            ax = axes[row, col]
            ax.imshow(data[index].T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
                      extent=(-0.5*spacing[1], (data.shape[1]-0.5)*spacing[1], -0.5*spacing[2], (data.shape[2]-0.5)*spacing[2]), interpolation='nearest')
            draw_ribbon_intersections(ax, surfaces, 0, index, [1, 2], spacing[[1, 2]])
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
    fig.suptitle(f'{subject} | MRI + hippocampal inner / middle / outer', color='white', fontsize=21, x=0.035, ha='left', y=0.98)
    add_ribbon_legend(fig)
    fig.text(0.035, 0.027, '2k hippocampal surfaces | Exact slice intersections | Original crop and slices', color='#cccccc', fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.075, top=0.885, wspace=0.06, hspace=0.16)
    fig.savefig(output / f'{subject}_sagittal_ribbon_surfaces.png', dpi=170, facecolor='black')
    plt.close(fig)






def render_coronal_ribbon(input_root, subject, output):
    """Render inner, middle, and outer intersections in the matching approved MRI layout."""
    base = input_root / f'{subject}'
    image = nib.load(base / 'anat' / f'{subject}_desc-preproc_T1w.nii.gz')
    data = image.get_fdata(dtype=np.float32)
    hemisphere_surfaces, vertex_arrays = [], []
    for hemi in ['L', 'R']:
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_midthickness.surf.gii')
        vertices, faces = mesh.darrays[0].data, mesh.darrays[1].data
        vertex_arrays.append(vertices)
        hemisphere_surfaces.append(load_ribbon_surfaces(base, subject, hemi, np.eye(4)))
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
        for surfaces in hemisphere_surfaces:
            draw_ribbon_intersections(ax, surfaces, 1, y, [0, 2], np.ones(2))
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'{panel+1:02d} | y = {y:.1f} mm', color='white', fontsize=12, pad=6)
        ax.text(0.015, 0.5, 'L', transform=ax.transAxes, color='white', fontsize=11)
        ax.text(0.96, 0.5, 'R', transform=ax.transAxes, color='white', fontsize=11)
        ax.plot([x[0]+2, x[0]+7], [z[0]+1.5]*2, color='white', lw=2)
        ax.text(x[0]+4.5, z[0]+2.3, '5 mm', color='white', ha='center', fontsize=8)
    fig.suptitle(f'{subject} | Coronal MRI + hippocampal inner / middle / outer', x=0.025, ha='left', y=0.98, color='white', fontsize=23)
    add_ribbon_legend(fig)
    fig.text(0.025, 0.026, 'Anterior to posterior, left to right then next row | L on left | 2k hippocampal surfaces | Exact slice intersections', color='#cccccc', fontsize=11)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.88, bottom=0.065, wspace=0.08, hspace=0.18)
    fig.savefig(output / f'{subject}_coronal_ribbon_surfaces.png', dpi=170, facecolor='black')
    plt.close(fig)
