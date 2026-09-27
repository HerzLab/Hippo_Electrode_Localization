"""Purpose: Display saved contact coverage and distances on individual hippocampal surfaces.

Author: Qirui Zhang
Created: 2026-09-21
Uses the existing HippoMaps geometry preparation and BrainSpace four-view convention.
Adapted for the standalone library: 2026-09-27.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from brainspace.plotting import plot_surf


def render_contact_map(surfaces, values, render_path, output_path, subject, title, binary=False, radius_mm=5.0):
    """Render fixed vertex values with shared scales and explicit uncovered regions."""
    uncovered_color = "#bfbfbf"
    cmap = ListedColormap([uncovered_color, "#0072b2"]) if binary else "viridis"
    limits = (0, 1) if binary else (0, radius_mm)
    for key, surface in surfaces.items():
        surface.append_array(values[key[0]].astype(np.float64), name="contact_display", at="point")
    plot_surf(surfaces, [["Lf", "Lu", "Ru", "Rf"]], array_name="contact_display",
              cmap=cmap, color_range=limits, share="both", zoom=0.95,
              nan_color=(0.75, 0.75, 0.75, 1), size=(1800, 440), interactive=False,
              offscreen=True, screenshot=True, filename=str(render_path), transparent_bg=False,
              mapper__interpolateScalarsBeforeMapping=False, suppress_warnings=True)
    fig = plt.figure(figsize=(16, 4.7), facecolor="white")
    ax = fig.add_axes([0.02, 0.20, 0.88, 0.60])
    ax.imshow(plt.imread(render_path))
    ax.set_axis_off()
    fig.suptitle(f"{subject} | {title}", x=0.04, ha="left", fontsize=19, fontweight="bold", y=0.97)
    bounds = ax.get_position()
    for column, label in enumerate(["Left folded", "Left unfolded", "Right unfolded", "Right folded"]):
        fig.text(bounds.x0 + bounds.width * (column + 0.5) / 4, 0.84, label, ha="center", fontsize=12)
    if binary:
        handles = [Patch(facecolor=uncovered_color, label="Not covered (0)"),
                   Patch(facecolor="#0072b2", label="Covered by at least one contact (1)")]
    else:
        bar_ax = fig.add_axes([0.927, 0.30, 0.012, 0.38])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, radius_mm), cmap=cmap), cax=bar_ax,
                     ticks=np.linspace(0, radius_mm, 6), label="Distance (mm)")
        handles = [Patch(facecolor=uncovered_color, label="Not covered")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, 0.08), ncol=2, frameon=False, fontsize=11)
    fig.text(0.04, 0.025, f"Individual 2k midthickness | Native T1w Euclidean distance <= {radius_mm:g} mm | No smoothing or extrapolation",
             fontsize=9, color="#444444")
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)
