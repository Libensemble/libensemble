"""
Plots a Gantt chart that maps each point in the history to a worker

Copy to a regression test and run:
    python process_history_to_make_chart.py <hist_filename.npy> <num>
where <hist_filename.npy> is the saved history file and <num> is the number of workers.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Declaring a figure "gnt"
fig, gnt = plt.subplots()
gen_work_color = "tab:orange"
sim_work_color = "tab:red"
man_work_color = "tab:blue"

filename = sys.argv[1]
H = np.load(filename)
num_workers = int(sys.argv[2])

H = H[H["sim_ended"]]
mint = np.min(H["gen_started_time"])
maxt = np.max(H["sim_ended_time"])

# Setting Y-axis limits, ticks, and labels
gnt.set_ylim(-0.1, num_workers + 1)
gnt.set_yticks(np.arange(0.5, num_workers + 1.5))
labels = ["man"] + ["w" + str(i) for i in range(1, num_workers + 1)]
gnt.set_yticklabels(labels)

# Setting X-axis limits
gnt.set_xlim(0, maxt - mint)

# Setting labels for x-axis and y-axis
gnt.set_xlabel("Seconds since start")

# Setting graph attribute
# gnt.grid(True)

for i in range(1, num_workers + 1):
    inds = np.where(H["gen_worker"] == i)[0]
    b = [(H[i]["gen_started_time"] - mint, H[i]["gen_ended_time"] - H[i]["gen_started_time"]) for i in inds]
    # Note: broken_barh takes (start, duration) not (start, end) pairs!
    gnt.broken_barh(b, (i + 0.1, 0.35), facecolors=gen_work_color, edgecolor=gen_work_color)

    inds = np.where(H["sim_worker"] == i)[0]
    b = [(H[i]["sim_started_time"] - mint, H[i]["sim_ended_time"] - H[i]["sim_started_time"]) for i in inds]
    gnt.broken_barh(b, (i + 0.55, 0.35), facecolors=sim_work_color, edgecolor=sim_work_color)

gnt.legend(["Gen work", "Sim work"], loc="lower right")
plt.savefig("gantt1.png", dpi=400)
