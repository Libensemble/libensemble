Running libEnsemble
===================

libEnsemble runs using a Manager/Worker paradigm. In most cases, one manager and multiples workers.
Each worker may run either a generator or simulator function (both are Python scripts). Generators
determine the parameters/inputs for simulations. Simulator functions run simulations, which often
involve running a user application from the Worker (see :doc:`Executor<executor/ex_index>`).

To use libEnsemble, you will need a calling script, which in turn will specify generator and
simulator functions. Many :doc:`examples<examples/examples_index>` are available.

There are currently three communication options for libEnsemble (determining how the Manager
and Workers communicate). These are ``mpi``, ``local``, ``tcp``. The default is ``mpi``.

.. note::
    You do not need the ``mpi`` communication mode to use the
    :doc:`MPI Executor<executor/mpi_executor>` to launch user MPI applications from workers.
    The communications modes described here only refer to how the libEnsemble manager and
    workers communicate.

MPI Comms
---------

This option uses ``mpi4py`` for the Manager/Worker communication. It is used automatically if
you run your libEnsemble calling script with an MPI runner. E.g::

    mpirun -np N python myscript.py

where ``N`` is the number of processors. This will launch one manager and
``N-1`` workers.

This option requires ``mpi4py`` to be installed to interface with the MPI on your system.
It works on a standalone system, and with both
:doc:`central and distributed modes<platforms/platforms_index>` of running libEnsemble on
multi-node systems.

It also potentially scales the best when running with many workers on HPC systems.

Limitations of MPI mode
^^^^^^^^^^^^^^^^^^^^^^^

If you are launching MPI applications from workers, then MPI is being nested. This is not
supported with Open MPI. This can be overcome by using a proxy launcher
(see :doc:`Balsam<executor/balsam_2_executor>`). This nesting does work, however,
with MPICH and its derivative MPI implementations.

It is also unsuitable to use this mode when running on the **launch** nodes of three-tier
systems (e.g. Theta/Summit). In that case ``local`` mode is recommended.

Local Comms
-----------

This option uses Python's built-in multiprocessing module for the manager/worker communications.
The ``comms`` type ``local`` and number of workers ``nworkers`` may be provided in the
:ref:`libE_specs<datastruct-libe-specs>` dictionary. Your calling script can then be run::

    python myscript.py

Alternatively, if your calling script uses the :doc:`parse_args()<utilities>` function
you can specify these on the command line::

    python myscript.py --comms local --nworkers N

where ``N`` is the number of workers. This will launch one manager and
``N`` workers.

libEnsemble will run on one node in this scenario. If the user wants to dedicate the node
to just the libEnsemble manager and workers, the ``libE_specs['dedicated_mode']`` option
can be set (see :doc:`central mode<platforms/platforms_index>`).

This mode is often used to run on a **launch** node of a three-tier
system (e.g. Theta/Summit), allowing the whole node allocation for
worker-launched application runs. In this scenario, make sure there are
no imports of ``mpi4py`` in your Python scripts.

Note that on macOS (since Python 3.8) and Windows, the default multiprocessing method is ``"spawn"`` instead
of ``"fork"``; to resolve many related issues, we recommend placing calling script code in
an ``if __name__ == "__main__":`` block.

Limitations of local mode
^^^^^^^^^^^^^^^^^^^^^^^^^

- You cannot run in :doc:`distributed mode<platforms/platforms_index>` on multi-node systems.
- In some scenarios, any import of ``mpi4py`` will cause this to break.
- It does not have the potential scaling of MPI mode, but is sufficient for most users.

TCP Comms
---------

The TCP option can be used to run the Manager on one system and launch workers to remote
systems or nodes over TCP. The necessary configuration options can be provided through
``libE_specs``, or on the command line if you are using the :doc:`parse_args()<utilities>` function.

The ``libE_specs`` options for TCP are::

    'comms' [string]:
        'tcp'
    'nworkers' [int]:
        Number of worker processes to spawn
    'workers' list:
        A list of worker hostnames.
    'ip' [String]:
        IP address
    'port' [int]:
        Port number.
    'authkey' [String]:
        Authkey.

Reverse-ssh interface
^^^^^^^^^^^^^^^^^^^^^

By specifying ``--comms ssh`` on the command line, libEnsemble workers can
be launched to remote ssh-accessible systems without needing to specify ``'port'`` or ``'authkey'``. This allows users
to colocate workers, simulation, or generator functions, and any applications they submit on the same machine. Such user
functions can also be persistent, unlike when launching remote functions via :ref:`funcX<funcx_ref>`.

The working directory and Python to run on the remote system need to be specified. Running a calling script may resemble::

    python myscript.py --comms ssh --workers machine1 machine2 --worker_pwd /home/workers --worker_python /home/.conda/.../python

.. note::
    Setting up public-key authentication on the worker host systems is recommended to avoid entering passwords.

.. note::
    This interface assumes that all remote machines share a filesystem. We'll be adjusting this in the future.

Limitations of TCP mode
^^^^^^^^^^^^^^^^^^^^^^^

- There cannot be two calls to ``libE`` in the same script.

Further command line options
----------------------------

See the **parse_args()** function in :doc:`Convenience Tools<utilities>` for further command line options.

.. _liberegister:

liberegister / libesubmit
-------------------------

libEnsemble now features a pair of command-line utilities for preparing and launching libEnsemble workflows onto almost
any machine and any scheduler, using a `PSI/J`_ Python implementation. This is an alternative approach
to maintaining system or scheduler-specific batch submission scripts.

- `liberegister`

Creates an initial, platform-independent PSI/J serialization of a libEnsemble submission. Run this utility on
a calling script in a familiar manner::

    liberegister my_calling_script.py --comms local --nworkers 4

This produces an initial `my_calling_script.json` serialization conforming to PSI/J's specification:

.. container:: toggle

    .. container:: header

       `my_calling_script.json`

    .. code-block:: JSON

        {
            "version": 0.1,
            "type": "JobSpec",
            "data": {
                "name": "libe-job",
                "executable": "python",
                "arguments": [
                    "my_calling_script.py",
                    "--comms",
                    "local",
                    "--nworkers",
                    "4"
                ],
                "directory": null,
                "inherit_environment": true,
                "environment": {
                    "PYTHONNOUSERSITE": "1"
                },
                "stdin_path": null,
                "stdout_path": null,
                "stderr_path": null,
                "resources": {
                    "node_count": 1,
                    "process_count": null,
                    "process_per_node": null,
                    "cpu_cores_per_process": null,
                    "gpu_cores_per_process": null,
                    "exclusive_node_use": true
                },
                "attributes": {
                    "duration": "30",
                    "queue_name": null,
                    "project_name": null,
                    "reservation_id": null,
                    "custom_attributes": {}
                },
                "launcher": null
            }
        }

- `libesubmit`

Further parameterizes a serialization, and submits a corresponding Job to the specified scheduler.
Running ``qsub``, ``sbatch``, etc. on some batch submission script is not needed. For instance::

    libesubmit my_calling_script.json -q debug -A project -s slurm --nnodes 8

Results in::

    *** libEnsemble 0.9.3 ***
    Imported PSI/J serialization: my_calling_script.json. Preparing submission...
    Calling script: my_calling_script.py
    ...found! Proceeding.
    Submitting Job!: Job[id=ce4ead75-a3a4-42a3-94ff-c44b3b2c7e61, native_id=None, executor=None, status=JobStatus[NEW, time=1658167808.5125017]]

    $ squeue --long --users=user
    Mon Jul 18 13:10:15 2022
             JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
           2508936    debug  ce4ead75     user  PENDING       0:00     30:00      8 (Priority)

This also produces a Job-specific representation, e.g:

.. container:: toggle

    .. container:: header

       `8ba9de56.my_calling_script.json`

    .. code-block:: JSON

        {
            "version": 0.1,
            "type": "JobSpec",
            "data": {
                "name": "libe-job",
                "executable": "/Users/jnavarro/miniconda3/envs/libe/bin/python3.8",
                "arguments": [
                    "my_calling_script.py",
                    "--comms",
                    "local",
                    "--nworkers",
                    "4"
                ],
                "directory": "/home/user/libensemble/scratch",
                "inherit_environment": true,
                "environment": {
                    "PYTHONNOUSERSITE": "1"
                },
                "stdin_path": null,
                "stdout_path": "8ba9de56.my_calling_script.out",
                "stderr_path": "8ba9de56.my_calling_script.err",
                "resources": {
                    "node_count": 8,
                    "process_count": null,
                    "process_per_node": null,
                    "cpu_cores_per_process": null,
                    "gpu_cores_per_process": null,
                    "exclusive_node_use": true
                },
                "attributes": {
                    "duration": "30",
                    "queue_name": "debug",
                    "project_name": "project",
                    "reservation_id": null,
                    "custom_attributes": {}
                },
                "launcher": null
            }
        }

If libesubmit is run on a ``.json`` serialization from liberegister and can't find the
specified calling script, it'll help search for matching candidate scripts.

.. _PSI/J: https://exaworks.org/psij

Persistent Workers
------------------
.. _persis_worker:

In a regular (non-persistent) worker, the user's generator or simulation function is called whenever the worker
receives work. A persistent worker is one that continues to run the generator or simulation function between work units,
maintaining the local data environment.

A common use-case consists of a persistent generator (such as :doc:`persistent_aposmm<examples/gen_funcs>`)
that maintains optimization data, while generating new simulation inputs. The persistent generator runs
on a dedicated worker while in persistent mode. This requires an appropriate
:doc:`allocation function<examples/alloc_funcs>` that will run the generator as persistent.

When running with a persistent generator, it is important to remember that a worker will be dedicated
to the generator and cannot run simulations. For example, the following run::

    mpirun -np 3 python my_script.py

would run one manager process, one worker with a persistent generator, and one worker running simulations.

If this example was run as::

    mpirun -np 2 python my_script.py

No simulations will be able to run.

Environment Variables
---------------------

Environment variables required in your run environment can be set in your Python sim or gen function.
For example::

    os.environ["OMP_NUM_THREADS"] = 4

set in your simulation script before the Executor submit command will export the setting to your run.

Further run information
-----------------------

For running on multi-node platforms and supercomputers, there are alternative ways to configure
libEnsemble to resources. See the :doc:`Running on HPC Systems<platforms/platforms_index>`
guide for more information, including some examples for specific systems.
