#!/usr/bin/env python

"""User tasks utilization plot

Script to produce utilisation plot based on how many workers are running user
tasks (submitted via a libEnsemble executor) at any given time. This does not
account for resource used by each task. The plot is written to a file.

This plot is produced from the libE_stats.txt file when the option
libE_stats['stats_fmt'] = {"task_datetime": True} is used.

The range of time is determined by the earliest task start and latest task
finish and so does not include any overhead before or after these times.

"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Basic options ---------------------------------------------------------------

infile = "libE_stats.txt"

sampling_freq = "1S"  # 1 second (default)
# sampling_freq = '10L'  # 10 microseconds - for very short simulations
# sampling_freq = '1M'   # 1 minute - for long simulations

# -----------------------------------------------------------------------------

# Produce start and end times for each task in each row (user function call).
sindex = 0
eindex = 0
run_stats = []
first_starts = []
last_ends = []
with open(infile) as f:
    for line in f:
        tstarts = []
        tends = []
        lst = line.split()
        sindex = 0
        eindex = 0
        done_index = 0
        for i, val in enumerate(lst):
            if val == "Tstart:":
                startdate = lst[i + 1]
                starttime = lst[i + 2]
                sindex += 1
            if val == "Tend:":
                enddate = lst[i + 1]
                endtime = lst[i + 2]
                eindex += 1
            if sindex > done_index and sindex == eindex:
                # Convert to pandas datetime so can compare
                start = pd.to_datetime(startdate + " " + starttime)
                end = pd.to_datetime(enddate + " " + endtime)
                tstarts.append(start)
                tends.append(end)
                done_index += 1

        if tstarts:
            first_starts.append(tstarts[0])

        if tends:
            last_ends.append(tends[-1])

        run_datetime = {"starts": tstarts, "ends": tends}
        run_stats.append(run_datetime)


# Find earliest task start and latest task end times to determine range.
df_fstarts = pd.DataFrame(first_starts)
df_lends = pd.DataFrame(last_ends)

time_start = first_starts[df_fstarts.index.min()]
time_end = last_ends[df_lends.index.max()]

# Split the range by sampling frequency
date_series = pd.date_range(time_start, time_end, freq=sampling_freq)

# For each sim or gen (user function call), determine if the sampled time falls between
# any of the task's start and end times.
counts = []
for i in range(len(date_series)):
    # Inclusive/exclusive to avoid multiple accounting
    count = 0
    ts = date_series[i]
    for row in run_stats:
        # For each row, is the sample time in any of the time intervals
        for ti, task in enumerate(row["starts"]):
            if task <= ts < row["ends"][ti]:
                count += 1
                break

    counts.append(count)

# Construct the graph
df_list = pd.DataFrame(date_series, columns=["datetime"])

# df_count = pd.DataFrame([counts], columns=['count']) # List goes to single row by default
df_count = pd.DataFrame({"count": counts})  # Transpose to columns like this

final = df_list.join(df_count)
final.plot(x="datetime", y="count", legend=None)
plt.xlabel("Time", fontsize=12)

plt.ylabel("Workers running user tasks", fontsize=12)
plt.ylim(ymin=0)  # To start graph at zero

# plt._show()
plt.savefig("tasks_util_v_time.png")
# plt.savefig("tasks_util_v_time.png", bbox_inches="tight", transparent=True)
