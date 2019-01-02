================
Analysis scripts
================

Note that all scripts produce a file rather than opening a plot interactively.

The following scripts must be run in the directory with **libe_summary.txt** file and extract plot informatin from that file. The information currently only pertains to the time spent in user sim or gen functions (not just the submission of jobs within those). 

* **plot_libe_calcs_util_v_time.py**: Extract worker utilization v time plot (with one second sampling). Shows number of workers running user sim or gen functions at any time.

* **plot_libE_histogram.py**: Create histogram showing the number of completed/killed user calculations binned by run-time.
