"""Purpose: Rasterize physical contact spheres and render coronal localization QC.

Author: Qirui Zhang
Created: 2026-09-18
Sphere geometry adapted from tmp/electrode_spheres_validation/Step1_WriteElectrodeMasks.py.
Adapted for the standalone library: 2026-09-27.
"""

import itertools
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates


def sphere_indices(center, affine, shape, radius_mm):
    """Return sphere voxel centers, including the count outside the native grid."""
    inverse = np.linalg.inv(affine)
    center_vox = nib.affines.apply_affine(inverse, center)
    extent = radius_mm * np.linalg.norm(inverse[:3, :3], axis=1)
    lower = np.floor(center_vox - extent).astype(int)
    upper = np.ceil(center_vox + extent).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(lower[d], upper[d] + 1)
                                 for d in range(3)], indexing='ij'), axis=-1).reshape(-1, 3)
    physical = nib.affines.apply_affine(affine, grid)
    sphere = grid[np.sum((physical - center) ** 2, axis=1) <= radius_mm ** 2]
    inside = sphere[((sphere >= 0) & (sphere < shape)).all(axis=1)]
    center_inside = bool(((center_vox >= -0.5) & (center_vox < shape - 0.5)).all())
    clipped = bool(((center_vox - extent < -0.5) |
                    (center_vox + extent > shape - 0.5)).any())
    return inside, center_vox, center_inside, clipped, len(sphere)


def save_native_image(data, reference, path, dtype, display_limits):
    """Preserve native geometry and write actual scaled intensities without rescaling."""
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data.astype(dtype), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    image.header.set_slope_inter(1, 0)
    image.header['cal_min'], image.header['cal_max'] = display_limits
    nib.save(image, path)


def render_coronal_mosaics(data, affine, centers, output_path, subject, window,
                           radius_mm=5.0):
    """Sample anatomical RAS coronal planes in one compact overlay-only mosaic.

    Colored circles are exact physical sphere intersections, not resampled masks.
    Crosses show centers within 1 mm of a plane, projected onto that plane.
    """
    corners = np.array(list(itertools.product(*[(-0.5, n - 0.5) for n in data.shape])))
    bounds = nib.affines.apply_affine(affine, corners)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    y_high = min(high[1], centers[:, 1].max() + radius_mm + 5.0)
    y_low = max(low[1], centers[:, 1].min() - radius_mm - 5.0)
    positions = np.linspace(y_high, y_low, 28)
    spacing_mm = (y_high - y_low) / 27
    inverse = np.linalg.inv(affine)

    # Find a shared display crop from visible anatomy and all sphere intersections.
    probe_x = np.arange(low[0], high[0] + 2, 2.0)
    probe_z = np.arange(low[2], high[2] + 2, 2.0)
    probe_xx, probe_zz = np.meshgrid(probe_x, probe_z)
    support = np.zeros(probe_xx.shape, dtype=bool)
    for y in positions:
        world = np.stack((probe_xx, np.full_like(probe_xx, y), probe_zz), axis=-1)
        vox = nib.affines.apply_affine(inverse, world).reshape(-1, 3).T
        plane = map_coordinates(data, vox, order=1, mode='constant', cval=0,
                                prefilter=False).reshape(probe_xx.shape)
        support |= plane > 0.05 * window[1]
        for center in centers[np.abs(centers[:, 1] - y) <= radius_mm]:
            support |= ((probe_xx - center[0]) ** 2 + (probe_zz - center[2]) ** 2
                        + (y - center[1]) ** 2 <= radius_mm ** 2)
    support_z, support_x = np.nonzero(support)
    low[[0, 2]] = [max(low[0], probe_x[support_x.min()] - 3),
                    max(low[2], probe_z[support_z.min()] - 3)]
    high[[0, 2]] = [min(high[0], probe_x[support_x.max()] + 3),
                     min(high[2], probe_z[support_z.max()] + 3)]
    columns, rows = 7, 4
    canvas_width, canvas_height = 1920, 1080
    header_height = 50
    tile_width = 260
    tile_height = (canvas_height - header_height) // rows
    pixels_mm = max((high[0] - low[0]) / (tile_width - 1),
                    (high[2] - low[2]) / (tile_height - 1))
    tile_height = int(np.ceil((high[2] - low[2]) / pixels_mm)) + 1
    x_mid = (low[0] + high[0]) / 2
    x = x_mid + (np.arange(tile_width) - (tile_width - 1) / 2) * pixels_mm
    z = high[2] - np.arange(tile_height) * pixels_mm
    xx, zz = np.meshgrid(x, z)
    crop_width = round(tile_width * 0.80)
    crop_top = round(tile_height * 0.05)
    crop_bottom = round(tile_height * 0.85)
    crop_height = crop_bottom - crop_top
    crop_left = (tile_width - crop_width) // 2
    crop_box = (crop_left, crop_top, crop_left + crop_width, crop_bottom)
    scale = min(canvas_width / (columns * crop_width),
                (canvas_height - header_height) / (rows * crop_height))
    display_width = int(np.floor(crop_width * scale))
    display_height = int(np.floor(crop_height * scale))
    left_margin = (canvas_width - columns * display_width) // 2
    top_margin = header_height + (canvas_height - header_height - rows * display_height) // 2
    inverse = np.linalg.inv(affine)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=12)
    page = Image.new('RGB', (canvas_width, canvas_height), 'black')
    draw = ImageDraw.Draw(page)
    draw.text((8, 4), f'{subject} | Native T1 coordinates | radius {radius_mm:g} mm | 28 coronal slices | 7 x 4', fill='white', font=font)
    draw.text((8, 29), f'Anterior to posterior, left to right then next row | spacing {spacing_mm:.2f} mm | L on left | Cyan: spheres; yellow: centers within 1 mm', fill='white', font=small)
    for slot, y in enumerate(positions):
        world = np.stack((xx, np.full_like(xx, y), zz), axis=-1)
        vox = nib.affines.apply_affine(inverse, world).reshape(-1, 3).T
        plane = map_coordinates(data, vox, order=1, mode='constant', cval=0,
                                prefilter=False).reshape(tile_height, tile_width)
        gray = np.clip((plane - window[0]) / (window[1] - window[0]), 0, 1)
        overlay = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        union = np.zeros(xx.shape, dtype=bool)
        for center in centers[np.abs(centers[:, 1] - y) <= radius_mm]:
            union |= ((xx - center[0]) ** 2 + (zz - center[2]) ** 2
                      + (y - center[1]) ** 2 <= radius_mm ** 2)
        overlay[union] = (0.55 * overlay[union] + 0.45 * np.array([0, 230, 255])).astype(np.uint8)
        tile = Image.fromarray(overlay)
        marker_draw = ImageDraw.Draw(tile)
        for center in centers[np.abs(centers[:, 1] - y) <= 1.0]:
            px = (center[0] - x[0]) / pixels_mm
            pz = (z[0] - center[2]) / pixels_mm
            marker_draw.line((px - 3, pz, px + 3, pz), fill='yellow', width=1)
            marker_draw.line((px, pz - 3, px, pz + 3), fill='yellow', width=1)
        tile = tile.crop(crop_box).resize((display_width, display_height), Image.Resampling.LANCZOS)
        marker_draw = ImageDraw.Draw(tile)
        marker_draw.text((6, 3), 'L', fill='white', font=small,
                         stroke_width=1, stroke_fill='black')
        marker_draw.text((display_width - 18, 3), 'R', fill='white', font=small,
                         stroke_width=1, stroke_fill='black')
        row, column = divmod(slot, columns)
        page.paste(tile, (left_margin + column * display_width, top_margin + row * display_height))
    page.save(output_path)
    return len(positions), 1
