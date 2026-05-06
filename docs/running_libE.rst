.. _running-libe:

Running libEnsemble
===================

.. note::
    You do not need the ``mpi`` communication mode to use the
    :doc:`MPI Executor<executor/ex_index>`. The communication modes described
    here only refer to how the libEnsemble manager and workers communicate.

Local Comms
-----------

Uses Python's built-in multiprocessing_ module.
The ``comms`` type ``local`` and number of workers ``nworkers`` for running simulators
may be provided in :ref:`libE_specs<datastruct-libe-specs>`.

Run:

    python myscript.py

Or, if the script uses the :meth:`parse_args<tools.parse_args>` function
or an :class:`Ensemble<libensemble.ensemble.Ensemble>` object with ``Ensemble(parse_args=True)``,
this can be specified on the command line:

    python myscript.py -n N

libEnsemble will run on **one node** in this scenario. To
:doc:`disallow this node<platforms/platforms_index>`
from app-launches (if running libEnsemble on a compute node),
set ``libE_specs["dedicated_mode"] = True``.

This mode can also be used to run on a **launch** node of a three-tier
system, ensuring the whole compute-node allocation is available for
launching apps. Make sure there are no imports of ``mpi4py`` in your Python scripts.

Note that on macOS and Windows, the default multiprocessing method is ``"spawn"``
instead of ``"fork"``; to resolve many related issues, we recommend placing
calling script code in an ``if __name__ == "__main__":`` block.

**Limitations of local mode**

- Workers cannot be :doc:`distributed<platforms/platforms_index>` across nodes.
- In some scenarios, any import of ``mpi4py`` will cause this to break.
- Does not have the potential scaling of MPI mode, but is sufficient for most users.

MPI Comms
---------

This option uses mpi4py_ for the Manager/Worker communication. It is used automatically if
you run your libEnsemble calling script with an MPI runner such as::

    mpirun -np N python myscript.py

where ``N`` is the number of processes. This will launch one manager and
``N-1`` simulator workers.

This option requires ``mpi4py`` to be installed to interface with the MPI on your system.
It works on a standalone system, and with both
:doc:`central and distributed modes<platforms/platforms_index>` of running libEnsemble on
multi-node systems.

It also potentially scales the best when running with many workers on HPC systems.

**Limitations of MPI mode**

If launching MPI applications from workers, then MPI is nested. **This is not
supported with Open MPI**. This can be overcome by using a proxy launcher.
This nesting does work with MPICH_ and its derivative MPI implementations.

It is also unsuitable to use this mode when running on the **launch** nodes of
three-tier systems. In that case ``local`` mode is recommended.

TCP Comms
---------

Run the Manager on one system and launch workers to remote
systems or nodes over TCP. Configure through
:class:`libE_specs<libensemble.specs.LibeSpecs>`, or on the command line
if using an :class:`Ensemble<libensemble.ensemble.Ensemble>` object with
``Ensemble(parse_args=True)``,

**Reverse-ssh interface**

Set ``comms`` to ``ssh`` to launch workers on remote ssh-accessible systems. This
co-locates workers, functions, and any applications. User
functions can also be persistent, unlike when launching remote functions via
:ref:`Globus Compute<globus_compute_ref>`.

The remote working directory and Python need to be specified. This may resemble::

    python myscript.py --comms ssh --workers machine1 machine2 --worker_pwd /home/workers --worker_python /home/.conda/.../python

**Limitations of TCP mode**

- There cannot be two calls to ``Ensemble.run()`` or ``libE()`` in the same script.

Further Command Line Options
----------------------------

See the :meth:`parse_args<tools.parse_args>` function in :doc:`Convenience Tools<utilities>` for
further command line options.

Environment Variables
---------------------

Environment variables required in your run environment can be set in your Python sim or gen function.
For example::

    os.environ["OMP_NUM_THREADS"] = 4

set in your simulation script before the Executor *submit* command will export the setting
to your run. For running a bash script in a sub environment when using the Executor, see
the ``env_script`` option to the :doc:`MPI Executor<executor/ex_index>`.

Running on Multi-Node Systems
-----------------------------

For running on multi-node platforms and supercomputers, there are alternative ways to configure
libEnsemble to resources. See the :doc:`Running on HPC Systems<platforms/platforms_index>`
guide for more information, including some examples for specific systems.

.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _MPICH: https://www.mpich.org/
.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html
