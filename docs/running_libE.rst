.. _running-libe:

Running libEnsemble
===================

Introduction
------------

libEnsemble runs with one manager and multiple workers. Each worker may run either
a generator or simulator function (both are Python scripts). Generators
determine the parameters/inputs for simulations. Simulator functions run and
manage simulations, which often involve running a user application (see
:doc:`Executor<executor/ex_index>`).

.. note::
    As of version 1.3.0, the generator can be run as a thread on the manager,
    using the :ref:`libE_specs<datastruct-libe-specs>` option **gen_on_manager**.
    When using this option, set the number of workers desired for running
    simulations. See :ref:`Running generator on the manager<gen-on-manager>`
    for more details.

To use libEnsemble, you will need a calling script, which in turn will specify
generator and simulator functions. Many :doc:`examples<examples/examples_index>`
are available.

There are currently three communication options for libEnsemble (determining how
the Manager and Workers communicate). These are ``local``, ``mpi``, ``tcp``.
The default is ``local`` if ``nworkers`` is specified, otherwise ``mpi``.

Note that ``local`` comms can be used on multi-node systems, where
the :doc:`MPI executor<executor/overview>` is used to distribute MPI applications
across the nodes. Indeed, this is the most commonly used option, even on large
supercomputers.

.. note::
    You do not need the ``mpi`` communication mode to use the
    :doc:`MPI Executor<executor/mpi_executor>`. The communication modes described
    here only refer to how the libEnsemble manager and workers communicate.

.. tab-set::

    .. tab-item:: Local Comms

        Uses Python's built-in multiprocessing_ module.
        The ``comms`` type ``local`` and number of workers ``nworkers`` may
        be provided in :ref:`libE_specs<datastruct-libe-specs>`.

        Then run::

            python myscript.py

        Or, if the script uses the :meth:`parse_args<tools.parse_args>` function
        or an :class:`Ensemble<libensemble.ensemble.Ensemble>` object with ``Ensemble(parse_args=True)``,
        you can specify these on the command line::

            python myscript.py --nworkers N

        This will launch one manager and ``N`` workers.

        The following abbreviated line is equivalent to the above::

            python myscript.py -n N

        libEnsemble will run on **one node** in this scenario. To
        :doc:`disallow this node<platforms/platforms_index>`
        from app-launches (if running libEnsemble on a compute node),
        set ``libE_specs["dedicated_mode"] = True``.

        This mode can also be used to run on a **launch** node of a three-tier
        system (e.g., Summit), ensuring the whole compute-node allocation is available for
        launching apps. Make sure there are no imports of ``mpi4py`` in your Python scripts.

        Note that on macOS (since Python 3.8) and Windows, the default multiprocessing method
        is ``"spawn"`` instead of ``"fork"``; to resolve many related issues, we recommend placing
        calling script code in an ``if __name__ == "__main__":`` block.

        **Limitations of local mode**

        - Workers cannot be :doc:`distributed<platforms/platforms_index>` across nodes.
        - In some scenarios, any import of ``mpi4py`` will cause this to break.
        - Does not have the potential scaling of MPI mode, but is sufficient for most users.

    .. tab-item:: MPI Comms

        This option uses mpi4py_ for the Manager/Worker communication. It is used automatically if
        you run your libEnsemble calling script with an MPI runner such as::

            mpirun -np N python myscript.py

        where ``N`` is the number of processes. This will launch one manager and
        ``N-1`` workers.

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
        three-tier systems (e.g., Summit). In that case ``local`` mode is recommended.

    .. tab-item:: TCP Comms

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

        - There cannot be two calls to ``libE()`` or ``Ensemble.run()`` in the same script.

Further Command Line Options
----------------------------

See the :meth:`parse_args<tools.parse_args>` function in :doc:`Convenience Tools<utilities>` for
further command line options.

Persistent Workers
------------------
.. _persis_worker:

In a regular (non-persistent) worker, the user's generator or simulation function is called
whenever the worker receives work. A persistent worker is one that continues to run the
generator or simulation function between work units, maintaining the local data environment.

A common use-case consists of a persistent generator (such as :doc:`persistent_aposmm<examples/gen_funcs>`)
that maintains optimization data while generating new simulation inputs. The persistent generator runs
on a dedicated worker while in persistent mode. This requires an appropriate
:doc:`allocation function<examples/alloc_funcs>` that will run the generator as persistent.

When running with a persistent generator, it is important to remember that a worker will be dedicated
to the generator and cannot run simulations. For example, the following run::

    mpirun -np 3 python my_script.py

starts one manager, one worker with a persistent generator, and one worker for running simulations.

If this example was run as::

    mpirun -np 2 python my_script.py

No simulations will be able to run.

.. _gen-on-manager:

Running generator on the manager
--------------------------------

The majority of libEnsemble use cases run a single generator. The
:ref:`libE_specs<datastruct-libe-specs>` option **gen_on_manager** will cause
the generator function to run on a thread on the manager. This can run
persistent user functions, sharing data structures with the manager, and avoids
additional communication to a generator running on a worker. When using this
option, the number of workers specified should be the (maximum) number of
concurrent simulations.

If modifying a workflow to use ``gen_on_manager`` consider the following.

* Set ``nworkers`` to the number of workers desired for running simulations.
* If using :meth:`add_unique_random_streams()<tools.add_unique_random_streams>`
  to seed random streams, the default generator seed will be zero.
* If you have a line like ``libE_specs["nresource_sets"] = nworkers -1``, this
  line should be removed.
* If the generator does use resources, ``nresource_sets`` can be increased as needed
  so that the generator and all simulations are resourced.

Environment Variables
---------------------

Environment variables required in your run environment can be set in your Python sim or gen function.
For example::

    os.environ["OMP_NUM_THREADS"] = 4

set in your simulation script before the Executor *submit* command will export the setting
to your run. For running a bash script in a sub environment when using the Executor, see
the ``env_script`` option to the :doc:`MPI Executor<executor/mpi_executor>`.

Further Run Information
-----------------------

For running on multi-node platforms and supercomputers, there are alternative ways to configure
libEnsemble to resources. See the :doc:`Running on HPC Systems<platforms/platforms_index>`
guide for more information, including some examples for specific systems.

.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _MPICH: https://www.mpich.org/
.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html
.. _PSI/J: https://exaworks.org/psij
