#!/usr/bin/env python

"""User function utilization plot

Script to produce utilization plot based on how many workers are running user
functions (sim or gens) at any given time. The plot is written to a file.

This plot is produced from the libE_stats.txt file which uses timings created
by the workers and so does not include manager/workers communications overhead.

The range of time is determined by the earliest user function start and latest
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

# Produce start and end times for each calculation (user function).
run_stats = []
with open(infile) as f:
    # content = f.readlines()
    for line in f:
        lst = line.split()
        foundstart = False
        foundend = False
        for i, val in enumerate(lst):
            if val == "Start:":
                startdate = lst[i + 1]
                starttime = lst[i + 2]
                foundstart = True
            if val == "End:":
                enddate = lst[i + 1]
                endtime = lst[i + 2]
                foundend = True
            if foundstart and foundend:
                run_datetime = {"start": startdate + " " + starttime, "end": enddate + " " + endtime}
                run_stats.append(run_datetime)
                break

df = pd.DataFrame(run_stats)

# Find earliest calculation start and latest end times to determine range.
df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])
df = df.sort_values(by="start")

time_start = df["start"][0]
dend = df.sort_values(by="end")
time_end = dend["end"].iloc[-1]

# Split the range by sampling frequency
date_series = pd.date_range(time_start, time_end, freq=sampling_freq)

# Determine if a user function is running at each sampling time.
counts = []
for i in range(len(date_series)):
    # Inclusive/exclusive to avoid multiple accounting - need high resolution
    count = sum((df["start"] <= date_series[i]) & (date_series[i] < df["end"]))
    counts.append(count)

# Construct the graph
df_list = pd.DataFrame(date_series, columns=["datetime"])

# df_count = pd.DataFrame([counts], columns=['count']) # List goes to single row by default
df_count = pd.DataFrame({"count": counts})  # Transpose to columns like this

final = df_list.join(df_count)

final.plot(x="datetime", y="count", legend=None, linewidth=2, fontsize=12)
# final.plot(x="datetime", y="count", legend=None)

plt.xlabel("Time", fontsize=14)
plt.ylabel("Active calculations", fontsize=14)
plt.ylim(ymin=0)  # To start graph at zero

# plt._show()
plt.savefig("calcs_util_v_time.png")
# plt.savefig("calcs_util_v_time.png", bbox_inches="tight", transparent=True)
