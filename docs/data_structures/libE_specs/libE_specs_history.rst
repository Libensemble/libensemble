History
=======

`Introduction <libE_specs.html>`__ \|\| `General <libE_specs_general.html>`__ \|\| `Directories <libE_specs_directories.html>`__ \|\| `Profiling <libE_specs_profiling.html>`__ \|\| `TCP <libE_specs_tcp.html>`__ \|\| **History** \|\| `Resources <libE_specs_resources.html>`__

**save_every_k_sims** [int]:
    Save history array to file after every k simulated points.

**save_every_k_gens** [int]:
    Save history array to file after every k generated points.

**save_H_and_persis_on_abort** [bool] = ``True``:
    Save states of ``H`` and ``persis_info`` to file on aborting after an exception.

**save_H_on_completion** [bool] = ``False``:
    Save state of ``H`` to file upon completing a workflow. Also enabled when either ``save_every_k_sims``
    or ``save_every_k_gens`` is set.

**save_H_with_date** [bool] = ``False``:
    ``H`` filename contains date and timestamp.

**H_file_prefix** [str] = ``"libE_history"``:
    Prefix for ``H`` filename.

**final_gen_send** [bool] = ``False``:
    Send final simulation results to persistent generators before shutdown.
    The results will be sent along with the ``PERSIS_STOP`` tag.
