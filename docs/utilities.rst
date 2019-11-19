Utilities
=========

libEnsemble features a utilities module to assist in writing consistent
calling scripts and user functions.


Utilities API
-------------

.. automodule:: utils
   :members:
   :no-undoc-members:

Examples
--------

Check inputs
~~~~~~~~~~~~

.. code-block:: python

    from libensemble.utils import check_inputs
    check_inputs(sim_specs=my_sim_specs, gen_specs=my_gen_specs, exit_criteria=ec)

Parse Args
~~~~~~~~~~

.. code-block:: python

    from libensemble.utils import parse_args
    nworkers, is_master, libE_specs, misc_args = parse_args()

From the shell::

    $ python calling_script --comms local --nworkers 4

Usage:

.. code-block:: bash

    usage: test_... [-h] [--comms [{local,tcp,ssh,client,mpi}]]
                    [--nworkers [NWORKERS]] [--workers WORKERS [WORKERS ...]]
                    [--workerID [WORKERID]] [--server SERVER SERVER SERVER]
                    [--pwd [PWD]] [--worker_pwd [WORKER_PWD]]
                    [--worker_python [WORKER_PYTHON]]
                    [--tester_args [TESTER_ARGS [TESTER_ARGS ...]]]

Save libE Output
~~~~~~~~~~~~~~~~

.. code-block:: python

    save_libE_output(H, persis_info, __file__, nworkers)

Add Unique Random Streams
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    persis_info = add_unique_random_streams(old_persis_info, nworkers + 1)
