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
If the argument ``dedicated_mode=True`` is used when initializing the Executor, then any node
that is running a libEnsemble manager or worker will be removed from the node-list available
to the workers, ensuring libEnsemble has dedicated nodes.

To run in central mode using a 5 node allocation with 4 workers. From the head node
of the allocation::

    mpirun -np 5 python myscript.py

or::

    python myscript.py --comms local --nworkers 4

Either of these will run libEnsemble (inc. manager and 4 workers) on the first node. The remaining
4 nodes will be divided amongst the workers for submitted applications. If the same run was
performed without ``dedicated_mode=True``, runs could be submitted to all 5 nodes. The number of workers
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
capability (eg. aprun, jsrun). MPI applications can only be submitted from these nodes. Examples
of these systems include: Summit, Sierra and Theta.

There are two ways of running libEnsemble on these kind of systems. The first, and simplest,
is to run libEnsemble on the launch nodes. This is often sufficient if the worker's sim or
gen scripts are not doing too much work (other than launching applications). This approach
is inherently centralized. The entire node allocation is available for the worker-launched
tasks.

To run libEnsemble on the compute nodes of these systems requires an alternative Executor,
such as :doc:`Balsam<../executor/balsam_executor>`, which runs on the
launch nodes and launches tasks submitted by workers. Running libEnsemble on the compute
nodes is potentially more scalable and will better manage ``sim_f`` and ``gen_f`` functions
that contain considerable computational work or I/O.

    .. image:: ../images/centralized_new_detailed_balsam.png
        :alt: central_balsam
        :scale: 40
        :align: center

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
using using :ref:`zero resource workers<zero_resource_workers>`.


Overriding Auto-detection
-------------------------

libEnsemble can automatically detect system information. This includes resource information, such as
available nodes and the number of cores on the node, and information about available MPI runners.

System detection for resources can be overridden using the :ref:`resource_info<resource_info>`
libE_specs option.

When using the MPI Executor, it is possible to override the detected information using the
`custom_info` argument. See the :doc:`MPI Executor<../executor/mpi_executor>` for more.

Instructions for Specific Platforms
-----------------------------------

The following subsections have more information about configuring and launching
libEnsemble on specific HPC systems.

.. toctree::
    :maxdepth: 2
    :titlesonly:

    bebop
    cori
    theta
    summit
    example_scripts

.. _Balsam: https://balsam.readthedocs.io/en/latest/
.. _Cooley: https://www.alcf.anl.gov/support-center/cooley
