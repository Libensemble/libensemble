=======================
Timing analysis scripts
=======================

Note that all plotting scripts produce a file rather than opening a plot
interactively.

The following scripts must be run in the directory with the ``libE_stats.txt``
file. They extract and plot information from that file.

* ``plot_libe_calcs_util_v_time.py``: Extracts worker utilization vs. time plot
  (with one-second sampling). Shows number of workers running user sim or gen
  functions over time.

* ``plot_libe_tasks_util_v_time.py``: Extracts launched task utilization v time
  plot (with one second sampling). Shows number of workers with active tasks,
  launched via the executor, over time.

* ``plot_libe_histogram.py``: Creates histogram showing the number of
  completed/killed/failed user calculations binned by run time.

========================
Results analysis scripts
========================

* ``print_npy.py``: Prints to screen from a given ``*.npy`` file containing a
  NumPy structured array. Use ``done`` to print only the lines containing
  ``'sim_ended'`` points. Example::

    ./print_npy.py run_libe_forces_results_History_length=1000_evals=8.npy done

* ``print_fields.py``: Prints to screen from a given ``*.npy`` file containing
  a NumPy structured array. This is a more versatile version of ``print_npy.py``
  that allows the user to select fields to print and boolean conditions determining
  which rows are printed (see ``./print_fields.py -h`` for usage).

* ``compare_npy.py``: Compares either two provided ``*.npy`` files or one
  provided ``*.npy`` file with an expected results file (by default located at
  ../expected.npy). A tolerance is given on floating-point results, and NANs are
  compared as equal. Variable fields (such as those containing a time) are
  ignored. These fields may need to be modified depending on the user's history
  array.

* ``plot_pareto_2d.py``: Loop through objective points in f and extract the Pareto
  front. Arguments are an ``*.npy`` file and a budget.

* ``plot_pareto_3d.py``: Loop through objective points in f and extract the Pareto
  front. Arguments are an ``*.npy`` file and a budget.

* ``print_pickle.py``: Prints to screen from a given ``*.pickle`` file. Example::

    ./print_pickle.py persis_info_length=1000_evals=1000_workers=2.pickle
