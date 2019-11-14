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
