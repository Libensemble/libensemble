#!/usr/bin/env python

"""Histogram of user function run-times (completed & killed).

Script to produce a histogram plot giving a count of user function (sim or
gen) calls by run-time intervals. Color shows completed versus killed versus
failed/exception.

This plot is produced from the libE_stats.txt file. Status is taken from the
calc_status returned by user functions.

The plot is written to a file.

"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Basic options ---------------------------------------------------------------

infile = "libE_stats.txt"
time_key = "Time:"
status_key = "Status:"
sim_only = True  # Ignore generator times
max_bins = 40
ran_ok = ["Completed"]  # List of OK states
run_killed = ["killed"]  # Searches for this word in status string
run_exception = ["Exception", "Failed"]

# -----------------------------------------------------------------------------


def search_for_keyword(in_list, kw_list):
    for i, val in enumerate(in_list):
        if val.endswith(":"):
            break  # New key word found
        else:
            if val in kw_list:
                return True
    return False


def append_to_list(mylst, glob_list, found_time):
    # Assumes Time comes first - else have to modify
    if found_time:
        mylst.append(glob_list[-1])
    else:
        print("Error Status found before time - exiting")
        sys.exit()


active_line_count = 0
in_times = []
in_times_ran = []
in_times_kill = []
in_times_exception = []


# Read straight from libEnsemble summary file.
with open(infile) as f:
    for line in f:
        lst = line.split()
        found_time = False
        found_status = False
        for i, val in enumerate(lst):
            if val == time_key:
                if sim_only and lst[i - 1] != "sim":
                    break
                in_times.append(lst[i + 1])
                found_time = True
            if val == status_key:
                if lst[i + 1] in ran_ok:
                    append_to_list(in_times_ran, in_times, found_time)  # Assumes Time comes first
                elif search_for_keyword(lst[i + 1 : len(lst)], run_killed):
                    append_to_list(in_times_kill, in_times, found_time)  # Assumes Time comes first
                elif search_for_keyword(lst[i + 1 : len(lst)], run_exception):
                    exceptions = True
                    append_to_list(in_times_exception, in_times, found_time)  # Assumes Time comes first
                else:
                    print(f"Error: Unknown status - rest of line: {lst[i + 1:len(lst)]}")
                    sys.exit()
                found_status = True
            if found_time and found_status:
                active_line_count += 1
                break

print(f"Processed {active_line_count} calcs")

times = np.asarray(in_times, dtype=float)
times_ran = np.asarray(in_times_ran, dtype=float)
times_kill = np.asarray(in_times_kill, dtype=float)
times_exc = np.asarray(in_times_exception, dtype=float)

num_bins = min(active_line_count, max_bins)
binwidth = (times.max() - times.min()) / num_bins
bins = np.arange(min(times), max(times) + binwidth, binwidth)

# Completed is the top segment
plt_times = [times_kill, times_ran]
labels = ["Killed", "Completed"]
colors = ["#FF7F0e", "#1F77B4"]  # Orange, Blue

# Insert at bottom (Dont want label if no exceptions)
if in_times_exception:
    plt_times.insert(0, times_exc)
    labels.insert(0, "Except/Failed")
    colors.insert(0, "#d62728")  # Red

plt.hist(plt_times, bins, edgecolor="black", linewidth=1.5, stacked=True, label=labels, color=colors)

if sim_only:
    calc = "sim"
else:
    calc = "calc"

titl = "Histogram of " + calc + " times" + " (" + str(active_line_count) + " user calcs)" + str(num_bins) + " bins"

plt.title(titl)
plt.xlabel("Calculation run time (sec)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis="y")
plt.legend(loc="best", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.show()
plt.savefig("hist_completed_v_killed.png")
# plt.savefig("hist_completed_v_killed.png", bbox_inches="tight", transparent=True)
