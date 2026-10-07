Cache
=====

:doc:`Introduction <libE_specs>` || :doc:`General <libE_specs_general>` || :doc:`Directories <libE_specs_directories>` || :doc:`Profiling <libE_specs_profiling>` || :doc:`History <libE_specs_history>` || :doc:`Resources <libE_specs_resources>` || **Cache**

**cache_long_sims** [bool] = ``False``:
    Cache simulation results with runtimes >1s to disk. Subsequent runs with the same
    configuration, callable code and state, VOCS, and H0 will access this cache.

    Upon the generator creating points already in the cache, those points will be skipped from
    being sent for evaluation. Instead the corresponding cached results are retrieved and returned
    to the generator.

    The cache is saved in ``cache_dir``. By default, its name uses the full SHA-256
    configuration hash.

**cache_tolerances** [dict[str, float]] = ``{}``:
    Optional absolute tolerances for matching cached numeric simulation inputs, keyed by
    history field name. Unspecified fields use exact matching; relative tolerance is zero.

**cache_dir** [str] = ``"~/.cache/libensemble"``:
    The directory to store the cache file. Defaults to ``~/.cache/libensemble``.

**cache_name** [str] = ``None``:
    Optional cache filename override, stored in ``cache_dir``. Cache contents are still
    validated against the full configuration hash.
