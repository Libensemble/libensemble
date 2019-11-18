Utilities
=========

libEnsemble features several modules and tools to assist in writing consistent
calling scripts and user functions.

Input consistency
-----------------

Users can check the formatting and consistency of ``exit_criteria`` and each
``specs`` dictionary with the ``check_inputs()`` function from the ``utils``
module. Provide any combination of these data structures as keyword arguments.
For example::

  from libensemble.utils import check_inputs
  check_inputs(sim_specs=my-sim_specs, gen_specs=my-gen_specs, exit_criteria=ec)


Save output to file
-------------------

The ``save_libE_output()`` function can to dump the contents of the History
array and ``persis_info`` to NumPy and pickle objects respectively following
a libEnsemble run. Import with::

    from libensemble.utils import save_libE_output

And write the output of a run to files::

    H, persis_info, _ = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                             libE_specs)

    if is_master:
        save_libE_output(H, persis_info, __file__, nworkers)


Per-worker random streams
-------------------------

Since many libEnsemble use-cases require that each worker accesses it's own random
stream, we provide a simple utility to populate the ``persis_info`` dictionary
with the necessary objects.

Import and run with::

    from libensemble.utils import add_unique_random_streams

    persis_info = add_unique_random_streams(old_persis_info, nworkers+1)


Parameters as command-line arguments
------------------------------------

The ``parse_args()`` function can be used to pass common libEnsemble parameters as
command-line arguments.

In your calling script::

    from libensemble.utils import parse_args
    nworkers, is_master, libE_specs, misc_args = parse_args()

From the shell, for example::

    $ python calling_script --comms local --nworkers 4

Usage:

.. code-block:: bash

    usage: test_... [-h] [--comms [{local,tcp,ssh,client,mpi}]]
                    [--nworkers [NWORKERS]] [--workers WORKERS [WORKERS ...]]
                    [--workerID [WORKERID]] [--server SERVER SERVER SERVER]
                    [--pwd [PWD]] [--worker_pwd [WORKER_PWD]]
                    [--worker_python [WORKER_PYTHON]]
                    [--tester_args [TESTER_ARGS [TESTER_ARGS ...]]]
