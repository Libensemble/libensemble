Profiling
=========

:doc:`Introduction <libE_specs>` || :doc:`General <libE_specs_general>` || :doc:`Directories <libE_specs_directories>` || **Profiling** || :doc:`History <libE_specs_history>` || :doc:`Resources <libE_specs_resources>`

**profile** [bool] = ``False``:
    Profile manager and worker logic using ``cProfile``.

**safe_mode** [bool] = ``False``:
    Prevents user functions from overwriting protected History fields, but requires moderate overhead.

**stats_fmt** [dict]:
    A dictionary of options for formatting ``"libE_stats.txt"``.
    See "Formatting Options for libE_stats.txt".

**live_data** [LiveData] = None:
    Add a live data capture object (e.g., for plotting).
