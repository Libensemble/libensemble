=======================
Timing analysis scripts
=======================

Note that all plotting scripts produce a file rather than opening a plot
interactively.

The following scripts must be run in the directory with the **libE_stats.txt**
file. They extract and plot information from that file.

* **plot_libe_calcs_util_v_time.py**: Extract worker utilization v time plot
  (with one second sampling). Shows number of workers running user sim or gen
  functions over time.

* **plot_libe_runs_util_v_time.py**: Extract launched job utilization v time
  plot (with one second sampling). Shows number of workers with active jobs,
  launched via the job controller, over time.

* **plot_libe_histogram.py**: Create histogram showing the number of
  completed/killed/failed user calculations binned by run-time.

========================
Results analysis scripts
========================

* **print_npy.py**: Prints to screen from a given ``*.npy`` file containing a
  NumPy structured array. Use ``done`` to only print the lines containing
  *returned* points. Example::

    ./print_npy.py run_libe_forces_results_History_length=1000_evals=8.npy done

* **compare_npy.py**: Compares either two provided ``*.npy`` files or one
  provided ``*.npy`` file with an expected results file (by default located at
  ../expected.npy). A tolerance is given on floating point results and NANs are
  compared as equal. Variable fields (such as those containing a time) are
  ignored. These fields may need to be modified depending on user's history
  array.
