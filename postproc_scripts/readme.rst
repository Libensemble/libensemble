================
Analysis scripts
================

Note that all scripts produce a file rather than opening a plot interactively.

The following scripts must be run in the directory with the **libE_stats.txt** file. They extract and plot information from that file.  

* **plot_libe_calcs_util_v_time.py**: Extract worker utilization v time plot (with one second sampling). Shows number of workers running user sim or gen functions over time.

* **plot_libe_runs_util_v_time.py**: Extract launched job utilization v time plot (with one second sampling). Shows number of workers with active jobs, launched via the job controller, over time.

* **plot_libE_histogram.py**: Create histogram showing the number of completed/killed/failed user calculations binned by run-time.
