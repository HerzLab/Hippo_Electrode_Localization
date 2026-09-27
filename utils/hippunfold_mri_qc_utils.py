"""Purpose: Render cropped MRI with midthickness and individual contact sphere edges.

Author: Qirui Zhang
Created: 2026-09-20
Adapted from the owner-reviewed midthickness_overlay previews.
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
from matplotlib.patches import Circle

def render_sagittal_overlay(input_root, subject, centers, output, radius_mm=5.0):
    """Render the approved MRI views without changing coordinates or contact eligibility."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.3), facecolor='black')
    for row, hemi in enumerate(['L', 'R']):
        base = input_root / f'{subject}'
        image = nib.as_closest_canonical(nib.load(base / 'anat' / f'{subject}_hemi-{hemi}_space-cropT1w_desc-preproc_T1w.nii.gz'))
        data = image.get_fdata(dtype=np.float32)
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_midthickness.surf.gii')
        vertices = nib.affines.apply_affine(np.linalg.inv(image.affine), mesh.darrays[0].data)
        faces = mesh.darrays[1].data
        spacing = np.array(image.header.get_zooms())
        bounds = np.vstack([vertices.min(0), vertices.max(0)])
        lower = np.maximum(0, np.floor(bounds[0] - 4 / spacing).astype(int))
        upper = np.minimum(image.shape, np.ceil(bounds[1] + 4 / spacing).astype(int) + 1)
        roi = data[tuple(slice(a, b) for a, b in zip(lower, upper))]
        vmin, vmax = np.percentile(roi[np.isfinite(roi) & (roi > 0)], [1, 99])
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel(), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        for col, fraction in enumerate([0.25, 0.5, 0.75]):
            index = int(round(bounds[0, 0] + fraction * np.diff(bounds[:, 0])[0]))
            plane = vtk.vtkPlane()
            plane.SetNormal(1, 0, 0)
            plane.SetOrigin(index, 0, 0)
            cutter = vtk.vtkCutter()
            cutter.SetInputData(poly)
            cutter.SetCutFunction(plane)
            cutter.Update()
            section = cutter.GetOutput()
            cut_points = vtk_to_numpy(section.GetPoints().GetData())
            segments = []
            for cell_index in range(section.GetNumberOfCells()):
                cell = section.GetCell(cell_index)
                coords = cut_points[[cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]]
                segments.append(coords[:, [1, 2]] * spacing[[1, 2]])
            ax = axes[row, col]
            ax.imshow(data[index].T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
                      extent=(-0.5*spacing[1], (data.shape[1]-0.5)*spacing[1], -0.5*spacing[2], (data.shape[2]-0.5)*spacing[2]), interpolation='nearest')
            yy, zz = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]), indexing='ij')
            voxel_centers = np.stack([np.full_like(yy, index), yy, zz], axis=-1)
            world_centers = nib.affines.apply_affine(image.affine, voxel_centers)
            nearest_squared = np.full(yy.shape, np.inf)
            for center in centers:
                squared = np.sum((world_centers - center) ** 2, axis=-1)
                nearest_squared = np.minimum(nearest_squared, squared)
            overlay = np.zeros((*yy.shape, 4), dtype=float)
            overlay[nearest_squared <= radius_mm ** 2] = [0.0, 0.85, 1.0, 0.35]
            overlay[nearest_squared <= min(1.0, 0.2 * radius_mm) ** 2] = [1.0, 0.15, 0.3, 1.0]
            ax.imshow(overlay.transpose(1, 0, 2), origin='lower',
                      extent=(-0.5*spacing[1], (data.shape[1]-0.5)*spacing[1], -0.5*spacing[2], (data.shape[2]-0.5)*spacing[2]), interpolation='nearest')
            # Trace each physical sphere separately, retaining arcs inside overlaps.
            for center in centers:
                squared = np.sum((world_centers - center) ** 2, axis=-1)
                if squared.min() < radius_mm ** 2 < squared.max():
                    ax.contour(yy * spacing[1], zz * spacing[2], squared,
                               levels=[radius_mm ** 2], colors=[(0, 0.85, 1, 0.65)], linewidths=0.85)
            ax.add_collection(LineCollection(segments, colors='#ff9900', linewidths=1.15))
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
    fig.suptitle(f'{subject} | MRI + midthickness + contacts', color='white', fontsize=21, x=0.035, ha='left', y=0.98)
    fig.text(0.035, 0.925, f'Orange: 2k midthickness | Cyan: {radius_mm:g} mm radius, 35% opacity | Red: {min(1.0, 0.2 * radius_mm):g} mm radius, solid', color='white', fontsize=12)
    fig.text(0.035, 0.027, 'Corrected T1 coordinates | Actual sphere-slice intersections | Display neighborhoods, not contact size | Original crop and slices', color='#cccccc', fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.075, top=0.885, wspace=0.06, hspace=0.16)
    fig.savefig(output / f'{subject}_sagittal_midthickness_contacts.png', dpi=170, facecolor='black')
    plt.close(fig)


def render_coronal_overlay(input_root, subject, centers, output, radius_mm=5.0):
    """Render the approved MRI views without changing coordinates or contact eligibility."""
    base = input_root / f'{subject}'
    image = nib.load(base / 'anat' / f'{subject}_desc-preproc_T1w.nii.gz')
    data = image.get_fdata(dtype=np.float32)
    surfaces, vertex_arrays = [], []
    for hemi in ['L', 'R']:
        mesh = nib.load(base / 'surf' / f'{subject}_hemi-{hemi}_space-T1w_den-2k_label-hipp_midthickness.surf.gii')
        vertices, faces = mesh.darrays[0].data, mesh.darrays[1].data
        vertex_arrays.append(vertices)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel(), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        surfaces.append(poly)
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
        nearest_squared = np.full(xx.shape, np.inf)
        for center in centers:
            squared = (xx-center[0])**2 + (y-center[1])**2 + (zz-center[2])**2
            nearest_squared = np.minimum(nearest_squared, squared)
        overlay = np.zeros((*xx.shape, 4))
        overlay[nearest_squared <= radius_mm ** 2] = [0, 0.85, 1, 0.35]
        overlay[nearest_squared <= min(1.0, 0.2 * radius_mm) ** 2] = [1, 0.15, 0.3, 1]
        ax.imshow(overlay, origin='lower', extent=extent, interpolation='nearest')
        for center in centers:
            plane_distance = abs(y-center[1])
            if plane_distance < radius_mm:
                section_radius = np.sqrt(radius_mm ** 2-plane_distance**2)
                ax.add_patch(Circle((center[0], center[2]), section_radius,
                                    fill=False, edgecolor=(0, 0.85, 1, 0.65), linewidth=0.85))
        segments = []
        for surface in surfaces:
            plane = vtk.vtkPlane()
            plane.SetOrigin(0, float(y), 0)
            plane.SetNormal(0, 1, 0)
            cutter = vtk.vtkCutter()
            cutter.SetInputData(surface)
            cutter.SetCutFunction(plane)
            cutter.Update()
            section = cutter.GetOutput()
            if section.GetNumberOfPoints() == 0:
                continue  # A hemisphere can end before the other within the shared AP extent.
            points = vtk_to_numpy(section.GetPoints().GetData())
            for i in range(section.GetNumberOfCells()):
                cell = section.GetCell(i)
                segments.append(points[[cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]][:, [0, 2]])
        ax.add_collection(LineCollection(segments, colors='#ff9900', linewidths=1.05))
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'{panel+1:02d} | y = {y:.1f} mm', color='white', fontsize=12, pad=6)
        ax.text(0.015, 0.5, 'L', transform=ax.transAxes, color='white', fontsize=11)
        ax.text(0.96, 0.5, 'R', transform=ax.transAxes, color='white', fontsize=11)
        ax.plot([x[0]+2, x[0]+7], [z[0]+1.5]*2, color='white', lw=2)
        ax.text(x[0]+4.5, z[0]+2.3, '5 mm', color='white', ha='center', fontsize=8)
    fig.suptitle(f'{subject} | Coronal MRI + midthickness + contacts', x=0.025, ha='left', y=0.98, color='white', fontsize=23)
    fig.text(0.025, 0.929, f'Orange: 2k midthickness | Cyan: {radius_mm:g} mm radius, 35% opacity | Red: {min(1.0, 0.2 * radius_mm):g} mm radius, solid', color='white', fontsize=13)
    fig.text(0.025, 0.026, 'Anterior to posterior, left to right then next row | L on left | Native T1, cropped to hippocampi | Actual sphere-plane intersections', color='#cccccc', fontsize=11)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.88, bottom=0.065, wspace=0.08, hspace=0.18)
    fig.savefig(output / f'{subject}_coronal_midthickness_contacts.png', dpi=170, facecolor='black')
    plt.close(fig)
