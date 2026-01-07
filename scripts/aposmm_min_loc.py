#!/usr/bin/env python3

"""
Plots the location of minima for each optimization run in 3D input space for
the N best runs.

To be run with both the history file (*.npy) and the persis_info file (*.pickle)
from a libEnsemble/APOSMM run present in the current directory. The most recent
of each file type present will be used for the plot.

"""

import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

N = 6  # number of opt runs to show.

x_name = "x0"
y_name = "x1"
z_name = "x2"

full_bounds = False  # For entire input space enter bounds below

if full_bounds:
    # Define parameter bounds
    mcr = 1e-2
    x0_min, x0_max = mcr, 1.0 - mcr
    x1_min, x1_max = -20.0, 20.0
    x2_min, x2_max = 1.0, 20.0

# Find the most recent .npy and pickle files
try:
    H_file = max(glob.glob("*.npy"), key=os.path.getmtime)
    persis_info_file = max(glob.iglob("*.pickle"), key=os.path.getctime)
except Exception:
    sys.exit("Need a *.npy and a *.pickle files in run dir. Exiting...")

H = np.load(H_file)

with open(persis_info_file, "rb") as f:
    index_sets = pickle.load(f)["run_order"]

# Filter best N opt runs for clearer graph
trim_sets = {key: indices[:-1] for key, indices in index_sets.items()}
min_f_per_set = [(key, indices, H["f"][indices].min()) for key, indices in trim_sets.items() if len(indices) > 0]
min_f_per_set_sorted = sorted(min_f_per_set, key=lambda x: x[2])[:N]

# Plotting
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

for key, indices, _ in min_f_per_set_sorted:
    min_f_index = indices[np.argmin(H["f"][indices])]

    # Extract the corresponding 3D x position from H
    try:
        x, y, z = H["x"][min_f_index]
    except ValueError:
        x = H[x_name][min_f_index]
        y = H[y_name][min_f_index]
        z = H[z_name][min_f_index]

    # Plot the 3D point
    ax.scatter(x, y, z, marker="o", s=50, label=f"Opt run {key}")

    # Draw a line from the point to the XY plane (z=0)
    ax.plot([x, x], [y, y], [0, z], color="grey", linestyle="--")

if full_bounds:
    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    ax.set_zlim(x2_min, x2_max)

# Label the plot
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel(z_name)
ax.set_title("Locations of best points from each optimization run")
ax.legend(bbox_to_anchor=(-0.1, 0.9), loc="upper left", borderaxespad=0)
plt.savefig(f"location_min_best{N}.png")
