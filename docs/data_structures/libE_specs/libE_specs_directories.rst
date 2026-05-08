Directories
===========

`Introduction <libE_specs.html>`__ \|\| `General <libE_specs_general.html>`__ \|\| **Directories** \|\| `Profiling <libE_specs_profiling.html>`__ \|\| `TCP <libE_specs_tcp.html>`__ \|\| `History <libE_specs_history.html>`__ \|\| `Resources <libE_specs_resources.html>`__

.. tab-set::

    .. tab-item:: General

        **use_workflow_dir** [bool] = ``False``:
            Whether to place *all* log files, dumped arrays, and default ensemble-directories in a
            separate ``workflow`` directory. Each run is suffixed with a hash.
            If copying back an ensemble directory from another location, the copy is placed here.

        **workflow_dir_path** [str]:
            Optional path to the workflow directory.

        **ensemble_dir_path** [str] = ``"./ensemble"``:
            Path to main ensemble directory. Can serve
            as single working directory for workers, or contain calculation directories.

            .. code-block:: python

                LibeSpecs.ensemble_dir_path = "/scratch/my_ensemble"

        **ensemble_copy_back** [bool] = ``False``:
            Whether to copy back contents of ``ensemble_dir_path`` to launch
            location. Useful if ``ensemble_dir_path`` is located on node-local storage.

        **reuse_output_dir** [bool] = ``False``:
            Whether to allow overwrites and access to previous ensemble and workflow directories in subsequent runs.
            ``False`` by default to protect results.

        **calc_dir_id_width** [int] = ``4``:
            The width of the numerical ID component of a calculation directory name. Leading
            zeros are padded to the sim/gen ID.

        **use_worker_dirs** [bool] = ``False``:
            Whether to organize calculation directories under worker-specific directories:

            .. tab-set::

                .. tab-item:: False

                    .. code-block::

                        - /ensemble_dir
                            - /sim0000
                            - /gen0001
                            - /sim0001
                            ...

                .. tab-item:: True

                    .. code-block::

                        - /ensemble_dir
                            - /worker1
                                - /sim0000
                                - /gen0001
                                - /sim0004
                                ...
                            - /worker2
                            ...

    .. tab-item:: Sims

        **sim_dirs_make** [bool] = ``False``:
            Whether to make calculation directories for each simulation function call.

        **sim_dir_copy_files** [list]:
            Paths to files or directories to copy into each sim directory, or ensemble directory.
            List of strings or ``pathlib.Path`` objects.

        **sim_dir_symlink_files** [list]:
            Paths to files or directories to symlink into each sim directory, or ensemble directory.
            List of strings or ``pathlib.Path`` objects.

        **sim_input_dir** [str]:
            Copy this directory's contents into the working directory upon calling the simulation function.
            Forms the base of a simulation directory.

    .. tab-item:: Gens

        **gen_dirs_make** [bool] = ``False``:
            Whether to make generator-specific calculation directories for each generator function call.
            *Each persistent generator creates a single directory*.

        **gen_dir_copy_files** [list]:
            Paths to copy into the working directory upon calling the generator function.
            List of strings or ``pathlib.Path`` objects

        **gen_dir_symlink_files** [list]:
            Paths to files or directories to symlink into each gen directory.
            List of strings or ``pathlib.Path`` objects

        **gen_input_dir** [str]:
            Copy this directory's contents into the working directory upon calling the generator function.
            Forms the base of a generator directory.
