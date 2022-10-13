Running on HPC Systems
======================

Central v Distributed
---------------------

libEnsemble has been developed, supported, and tested on systems of highly varying
scales, from laptops to thousands of compute nodes. On multi-node systems, there are
two basic modes of configuring libEnsemble to run and launch tasks (user applications)
on the available nodes.

The first mode we refer to as **central** mode, where the libEnsemble manager and worker processes
are grouped on to one or more dedicated nodes. Workers' launch applications on to
the remaining allocated nodes:

    .. image:: ../images/centralized_new_detailed.png
        :alt: centralized
        :scale: 30
        :align: center

Alternatively, in **distributed** mode, the libEnsemble (manager/worker) processes
will share nodes with submitted tasks. This enables libEnsemble, using the *mpi4py*
communicator, to be run with the workers spread across nodes so as to be co-located
with their tasks.

    .. image:: ../images/distributed_new_detailed.png
        :alt: distributed
        :scale: 30
        :align: center

Configurations with multiple nodes per worker or multiple workers per node are both
common use cases. The distributed approach allows the libEnsemble worker to read files
produced by the application on local node storage. HPC systems that allow only one
application to be launched to a node at any one time prevent distributed configuration.

Configuring the Run
-------------------

On systems with a job scheduler, libEnsemble is typically run within a single
:doc:`job submission<example_scripts>`. All user simulations will run on
the nodes within that allocation.

*How does libensemble know where to run tasks (user applications)?*

The libEnsemble :doc:`Executor<../executor/ex_index>` can be initialized from the user calling
script, and then used by workers to run tasks. The Executor will automatically detect the nodes
available on most systems. Alternatively, the user can provide a file called **node_list** in
the run directory. By default, the Executor will divide up the nodes evenly to each worker.
If the argument ``libE_specs['dedicated_mode']=True`` is used when initializing libEnsemble, then any node
that is running a libEnsemble manager or worker will be removed from the node-list available
to the workers, ensuring libEnsemble has dedicated nodes.

To run in central mode using a 5 node allocation with 4 workers. From the head node
of the allocation::

    mpirun -np 5 python myscript.py

or::

    python myscript.py --comms local --nworkers 4

Either of these will run libEnsemble (inc. manager and 4 workers) on the first node. The remaining
4 nodes will be divided amongst the workers for submitted applications. If the same run was
performed without ``libE_specs['dedicated_mode']=True``, runs could be submitted to all 5 nodes. The number of workers
can be modified to allow either multiple workers to map to each node or multiple nodes per worker.

To launch libEnsemble distributed requires a less trivial libEnsemble run script.
For example::

    mpirun -np 5 -ppn 1 python myscript.py

would launch libEnsemble with 5 processes across 5 nodes. However, the manager would have its
own node, which is likely wasteful. More often, a machinefile is used to add the manager to
the first node. In the :doc:`examples<example_scripts>` directory, you can find an example submission
script, configured to run libensemble distributed, with multiple workers per node or multiple nodes
per worker, and adding the manager onto the first node.

HPC systems that only allow one application to be launched to a node at any one time,
will not allow a distributed configuration.

Systems with Launch/MOM nodes
-----------------------------

Some large systems have a 3-tier node setup. That is, they have a separate set of launch nodes
(known as MOM nodes on Cray Systems). User batch jobs or interactive sessions run on a launch node.
Most such systems supply a special MPI runner which has some application-level scheduling
capability (eg. ``aprun``, ``jsrun``). MPI applications can only be submitted from these nodes. Examples
of these systems include: Summit, Sierra and Theta.

There are two ways of running libEnsemble on these kind of systems. The first, and simplest,
is to run libEnsemble on the launch nodes. This is often sufficient if the worker's simulation
or generation functions are not doing much work (other than launching applications). This approach
is inherently centralized. The entire node allocation is available for the worker-launched
tasks.

However, running libEnsemble on the compute nodes is potentially more scalable and
will better manage simulation and generation functions that contain considerable
computational work or I/O. Therefore the second option is to use proxy task-execution
services like Balsam_.

Balsam - Externally managed applications
----------------------------------------

Running libEnsemble on the compute nodes while still submitting additional applications
requires alternative Executors that connect to external services like Balsam_. Balsam
can take tasks submitted by workers and execute them on the remaining compute nodes,
or if using Balsam 2, *to entirely different systems*.

    .. figure:: ../images/centralized_new_detailed_balsam.png
        :alt: central_balsam
        :scale: 30
        :align: center

        Single-System: libEnsemble + LegacyBalsamMPIExecutor

    .. figure:: ../images/balsam2.png
        :alt: balsam2
        :scale: 40
        :align: center

        (New) Multi-System: libEnsemble + BalsamExecutor

As of v0.9.0, libEnsemble supports both "legacy" Balsam via the
:doc:`LegacyBalsamMPIExecutor<../executor/legacy_balsam_executor>`
and Balsam 2 via the :doc:`BalsamExecutor<../executor/balsam_2_executor>`.

Submission scripts for running on launch/MOM nodes and for using Balsam, can be be found in
the :doc:`examples<example_scripts>`.

Mapping Tasks to Resources
--------------------------

The :ref:`resource manager<resources_index>` can :ref:`detect system resources<resource_detection>`,
and partition these to workers. The :doc:`MPI Executor<../executor/mpi_executor>`
accesses the resources available to the current worker when launching tasks.

Zero-resource workers
~~~~~~~~~~~~~~~~~~~~~

Users with persistent ``gen_f`` functions may notice that the persistent workers
are still automatically assigned system resources. This can be resolved by
:ref:`fixing the number of resource sets<zero_resource_workers>`.

Overriding Auto-detection
-------------------------

libEnsemble can automatically detect system information. This includes resource information, such as
available nodes and the number of cores on the node, and information about available MPI runners.

System detection for resources can be overridden using the :ref:`resource_info<resource_info>`
libE_specs option.

When using the MPI Executor, it is possible to override the detected information using the
`custom_info` argument. See the :doc:`MPI Executor<../executor/mpi_executor>` for more.

 .. _funcx_ref:

funcX - Remote User functions
-----------------------------

*Alternatively to much of the above*, if libEnsemble is running on some resource with
internet access (laptops, login nodes, other servers, etc.), workers can be instructed to
launch generator or simulator user function instances to separate resources from
themselves via funcX_, a distributed, high-performance function-as-a-service platform:

    .. image:: ../images/funcxmodel.png
        :alt: running_with_funcx
        :scale: 50
        :align: center

This is useful for running ensembles across machines and heterogeneous resources, but
comes with several caveats:

    1. User functions registered with funcX must be *non-persistent*, since
       manager-worker communicators can't be serialized or used by a remote resource.

    2. Likewise, the ``Executor.manager_poll()`` capability is disabled. The only
       available control over remote functions by workers is processing return values
       or exceptions when they complete.

    3. funcX imposes a `handful of task-rate and data limits`_ on submitted functions.

    4. Users are responsible for authenticating via Globus_ and maintaining their
       `funcX endpoints`_ on their target systems.

Users can still define Executor instances within their user functions and submit
MPI applications normally, as long as libEnsemble and the target application are
accessible on the remote system::

    # Within remote user function
    from libensemble.executors import MPIExecutor
    exctr = MPIExecutor()
    exctr.register_app(full_path='/home/user/forces.x', app_name='forces')
    task = exctr.submit(app_name='forces', num_procs=64)

Specify a funcX endpoint in either ``sim_specs`` or ``gen_specs`` via the ``funcx_endpoint``
key. For example::

    sim_specs = {
        'sim_f': sim_f,
        'in': ['x'],
        'out': [('f', float)],
        'funcx_endpoint': '3af6dc24-3f27-4c49-8d11-e301ade15353',
    }

See the ``libensemble/tests/scaling_tests/funcx_forces`` directory for a complete
remote-simulation example.

Instructions for Specific Platforms
-----------------------------------

The following subsections have more information about configuring and launching
libEnsemble on specific HPC systems.

.. toctree::
    :maxdepth: 2
    :titlesonly:

    bebop
    cori
    perlmutter
    spock/crusher <spock_crusher>
    summit
    theta
    srun
    example_scripts

.. _Balsam: https://balsam.readthedocs.io/en/latest/
.. _Cooley: https://www.alcf.anl.gov/support-center/cooley
.. _funcX: https://funcx.org/
.. _`funcX endpoints`: https://funcx.readthedocs.io/en/latest/endpoints.html
.. _Globus: https://www.globus.org/
.. _`handful of task-rate and data limits`: https://funcx.readthedocs.io/en/latest/limits.html
