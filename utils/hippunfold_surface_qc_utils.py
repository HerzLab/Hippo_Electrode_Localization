"""Purpose: Render individual 2k surface QC figures using HippoMaps conventions.

Author: Qirui Zhang
Created: 2026-09-20
Geometry preparation adapted from HippoMaps 0.1.17 plotting.py.
Promoted from the approved individual_surface_preview pilot on 2026-09-20.
Adapted for the standalone library: 2026-09-27.
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_surf
from hippomaps.plotting import _apply_unfold_transforms, _rotate_in_place

LABEL_NAMES = ["Unassigned", "Subiculum", "CA1", "CA2", "CA3", "CA4", "DG"]
LABEL_COLORS = ["#d9d9d9", "#4477aa", "#66ccee", "#228833", "#ccbb44", "#ee6677", "#aa3377"]
LABEL_CMAP = ListedColormap(LABEL_COLORS)


def load_surfaces(root, subject, structure):
    """Read native and unfolded meshes without substituting a template."""
    surfaces = {}
    for hemi in ["L", "R"]:
        for space, suffix in [("T1w", "f"), ("unfold", "u")]:
            path = root / f"{subject}" / "surf" / f"{subject}_hemi-{hemi}_space-{space}_den-2k_label-{structure}_midthickness.surf.gii"
            surface = read_surface(str(path))
            if space == "unfold":
                surface = _apply_unfold_transforms(surface, "2k", structure, False, True, hemi)
            else:
                _rotate_in_place(surface, 20)
            surfaces[hemi + suffix] = surface
    return surfaces


def load_metric(root, subject, feature):
    """Read fixed HippUnfold v2 metric paths, retaining native vertex order."""
    values = {}
    for hemi in ["L", "R"]:
        prefix = f"{subject}_hemi-{hemi}_den-2k_label-hipp"
        if feature in ["AP", "PD"]:
            name = f"{subject}_dir-{feature}_hemi-{hemi}_den-2k_label-hipp_desc-laplace_coords.shape.gii"
        elif feature == "subfields":
            name = f"{prefix}_atlas-multihist7_subfields.label.gii"
        else:
            name = f"{prefix}_{feature}.shape.gii"
        values[hemi] = nib.load(root / f"{subject}" / "metric" / name).darrays[0].data.copy()
    return values


def render_row(surfaces, values, filename, cmap, limits):
    """Render four views with shared lookup limits, without clipping arrays."""
    for key, surface in surfaces.items():
        surface.append_array(values[key[0]].astype(np.float64), name="display", at="point")  # Preserve a fixed lookup table even when some labels are absent.
    plot_surf(surfaces, [["Lf", "Lu", "Ru", "Rf"]], array_name="display",
              cmap=cmap, color_range=limits, share="both", zoom=0.95,
              size=(1800, 440), interactive=False, offscreen=True,
              screenshot=True, filename=str(filename), transparent_bg=False,
              mapper__interpolateScalarsBeforeMapping=not isinstance(cmap, ListedColormap),
              suppress_warnings=True)


def assemble_figure(subject, rows, output, title, footer, categorical=False):
    """Compose rendered surfaces with readable labels and explicit color scales."""
    fig = plt.figure(figsize=(16, 1.25 + 3.45 * len(rows)), facecolor="white")
    grid = fig.add_gridspec(len(rows), 1, left=0.02, right=0.92, top=0.87,
                           bottom=0.10, hspace=0.20)
    fig.suptitle(f"{subject} | {title}", x=0.04, ha="left", fontsize=19, fontweight="bold", y=0.97)
    for index, row in enumerate(rows):
        image_path, label, cmap, limits = row
        ax = fig.add_subplot(grid[index])
        ax.imshow(plt.imread(image_path))
        ax.set_axis_off()
        ax.set_title(label, loc="left", fontsize=12, pad=3)
        if index == 0:
            bounds = ax.get_position()
            for column, text in enumerate(["Left folded", "Left unfolded", "Right unfolded", "Right folded"]):
                fig.text(bounds.x0 + bounds.width * (column + 0.5) / 4, 0.915, text, ha="center", fontsize=12)
        if not categorical:
            bounds = ax.get_position()
            bar_ax = fig.add_axes([0.936, bounds.y0 + bounds.height * 0.15, 0.011, bounds.height * 0.70])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*limits), cmap=cmap), cax=bar_ax)
    if categorical:
        handles = [Patch(facecolor=color, label=name) for name, color in zip(LABEL_NAMES, LABEL_COLORS)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=7, frameon=False, fontsize=11)
    fig.text(0.04, 0.022, footer, fontsize=9, color="#444444")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
