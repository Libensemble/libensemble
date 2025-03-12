#!/usr/bin/env python3

"""
Plot f by optimization run, for N best runs. Other points are shown in grey

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
plt.figure(figsize=(10, 6))
plt.scatter(range(len(H["f"])), H["f"], color="lightgrey", label="Other points", zorder=1)

num_runs = len(index_sets)
# for key, indices in index_sets.items():
for key, indices, _ in min_f_per_set_sorted:
    # Extract the 'f' values for each index set
    f_values = H["f"][indices]

    ratio = (f_values[-1] - f_values[-2]) / f_values[-2]
    # print(f"Tolerance of the last two values for {key}: {ratio}")

    plt.plot(indices, f_values, marker="o", label=f"Opt run {key}", zorder=2)

    # Identify the index and value of the minimum f
    min_index = indices[np.argmin(f_values)]
    min_f_value = np.min(f_values)

    # Mark the minimum f value with a distinct marker
    plt.scatter(min_index, min_f_value, color="red", edgecolor="black", s=50, zorder=3)

# Add a dummy point to the legend for "minima of opt run"
plt.scatter([], [], color="red", edgecolor="black", s=50, label="Best value of opt run")

plt.xlabel("Simulation ID")
plt.ylabel("Objective value")
plt.title(f"Objective values by Optimization runs. Best {N} runs from {num_runs}")
plt.legend()
plt.grid(True)
plt.savefig(f"opt_runs_best{N}.png")
