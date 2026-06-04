Profiling
=========

`Introduction <libE_specs.html>`__ \|\| `General <libE_specs_general.html>`__ \|\| `Directories <libE_specs_directories.html>`__ \|\| **Profiling** \|\| `TCP <libE_specs_tcp.html>`__ \|\| `History <libE_specs_history.html>`__ \|\| `Resources <libE_specs_resources.html>`__

**profile** [bool] = ``False``:
    Profile manager and worker logic using ``cProfile``.

**safe_mode** [bool] = ``False``:
    Prevents user functions from overwriting protected History fields, but requires moderate overhead.

**stats_fmt** [dict]:
    A dictionary of options for formatting ``"libE_stats.txt"``.
    See "Formatting Options for libE_stats.txt".

**live_data** [LiveData] = None:
    Add a live data capture object (e.g., for plotting).
