.. _platform-index:

Running on HPC Systems
======================

libEnsemble has been tested on systems of highly varying scales, from laptops to
thousands of compute nodes. On multi-node systems, there are a few alternative
ways of configuring libEnsemble to run and launch tasks (i.e., user applications)
on the available nodes.

The :doc:`Forces tutorial <../../tutorials/executor_forces_tutorial>` gives an
example with a simple MPI application.

Note that while the diagrams below show one application being run per node,
configurations with **multiple nodes per worker** or **multiple workers per node**
are both common use cases.

Centralized Running
-------------------

The default communications scheme places the manager and workers on the first node.
The :doc:`MPI Executor<../executor/mpi_executor>` can then be invoked by each
simulation worker, and libEnsemble will distribute user applications across the
node allocation. This is the **most common approach** where each simulation
runs an MPI application.

The generator will run on a worker by default, but if running a single generator,
the :ref:`libE_specs<datastruct-libe-specs>` option **gen_on_manager** is recommended,
which runs the generator on the manager (using a thread) as below.

.. list-table::
   :widths: 60 40

   * - .. image:: ../images/centralized_gen_on_manager.png
          :alt: centralized
          :scale: 55

     - In calling script:

       .. code-block:: python
          :linenos:

          ensemble.libE_specs = LibeSpecs(
              gen_on_manager=True,
          )

       A SLURM batch script may include:

       .. code-block:: bash

          #SBATCH --nodes 3

          python run_libe_forces.py --nworkers 3

When using **gen_on_manager**, set ``nworkers`` to the number of workers desired for running simulations.

Dedicated Mode
^^^^^^^^^^^^^^

If the :ref:`libE_specs<datastruct-libe-specs>` option **dedicated_mode** is set to
True, the MPI executor will not launch applications on nodes where libEnsemble Python
processes (manager and workers) are running. Workers launch applications onto the
remaining nodes in the allocation.

.. list-table::
   :widths: 60 40

   * - .. image:: ../images/centralized_dedicated.png
          :alt: centralized dedicated mode
          :scale: 30

     - In calling script:

       .. code-block:: python
          :linenos:

          ensemble.libE_specs = LibeSpecs(
              num_resource_sets=2,
              dedicated_mode=True,
          )

       A SLURM batch script may include:

       .. code-block:: bash

          #SBATCH --nodes 3

          python run_libe_forces.py --nworkers 3

Note that **gen_on_manager** is not set in the above example.

Distributed Running
-------------------

In the **distributed** approach, libEnsemble can be run using the **mpi4py**
communicator, with workers distributed across nodes. This is most often used
when workers run simulation code directly, via a Python interface. The user
script is invoked with an MPI runner, for example (using an `mpich`-based MPI)::

    mpirun -np 4 -ppn 1 python myscript.py

The distributed approach, can also be used with the executor, to co-locate workers
with the applications they submit. Ensuring that workers are placed as required in this
case requires :ref:`a careful MPI rank placement <slurm_mpi_distributed>`.

    .. image:: ../images/distributed_new_detailed.png
        :alt: distributed
        :scale: 30
        :align: center

This allows the libEnsemble worker to read files produced by the application on
local node storage.

Configuring the Run
-------------------

On systems with a job scheduler, libEnsemble is typically run within a single
:doc:`job submission<example_scripts>`. All user simulations will run on
the nodes within that allocation.

*How does libEnsemble know where to run tasks (user applications)?*

The libEnsemble :doc:`MPI Executor<../executor/mpi_executor>` can be initialized from the user calling
script, and then used by workers to run tasks. The Executor will automatically detect the nodes
available on most systems. Alternatively, the user can provide a file called **node_list** in
the run directory. By default, the Executor will divide up the nodes evenly to each worker.

Mapping Tasks to Resources
--------------------------

The :ref:`resource manager<resources_index>` detects node lists from
:ref:`common batch schedulers<resource_detection>`,
and partitions these to workers. The :doc:`MPI Executor<../executor/mpi_executor>`
accesses the resources available to the current worker when launching tasks.

Zero-resource workers
---------------------

Users with persistent ``gen_f`` functions may notice that the persistent workers
are still automatically assigned system resources. This can be resolved by using
the ``gen_on_manager`` option or by
:ref:`fixing the number of resource sets<zero_resource_workers>`.

Assigning GPUs
--------------

libEnsemble automatically detects and assigns Nvidia, AMD, and Intel GPUs without modifying the user scripts. This automatically works on many systems, but if the assignment is incorrect or needs to be modified the user can specify :ref:`platform information<datastruct-platform-specs>`.
The :doc:`forces_gpu tutorial<../tutorials/forces_gpu_tutorial>` shows an example of this.

Varying resources
-----------------

libEnsemble also features :ref:`dynamic resource assignment<var-resources-gpu>`, whereby the
number of processes and/or the number of GPUs can be a set for each simulation by the generator.

Overriding Auto-Detection
-------------------------

libEnsemble can automatically detect system information. This includes resource information, such as
available nodes and the number of cores on the node, and information about available MPI runners.

System detection for resources can be overridden using the :ref:`resource_info<resource_info>`
libE_specs option.

When using the MPI Executor, it is possible to override the detected information using the
`custom_info` argument. See the :doc:`MPI Executor<../executor/mpi_executor>` for more.

Systems with Launch/MOM Nodes
-----------------------------

Some large systems have a 3-tier node setup. That is, they have a separate set of launch nodes
(known as MOM nodes on Cray Systems). User batch jobs or interactive sessions run on a launch node.
Most such systems supply a special MPI runner that has some application-level scheduling
capability (e.g., ``aprun``, ``jsrun``). MPI applications can only be submitted from these nodes. Examples
of these systems include Summit and Sierra.

There are two ways of running libEnsemble on these kinds of systems. The first, and simplest,
is to run libEnsemble on the launch nodes. This is often sufficient if the worker's simulation
or generation functions are not doing much work (other than launching applications). This approach
is inherently centralized. The entire node allocation is available for the worker-launched
tasks.

However, running libEnsemble on the compute nodes is potentially more scalable and
will better manage simulation and generation functions that contain considerable
computational work or I/O. Therefore the second option is to use Globus Compute
to isolate this work from the workers.

.. _globus_compute_ref:

Globus Compute - Remote User Functions
--------------------------------------

If libEnsemble is running on some resource with
internet access (laptops, login nodes, other servers, etc.), workers can be instructed to
launch generator or simulator user function instances to separate resources from
themselves via `Globus Compute`_ (formerly funcX), a distributed, high-performance function-as-a-service platform:

    .. image:: ../images/funcxmodel.png
        :alt: running_with_globus_compute
        :scale: 50
        :align: center

This is useful for running ensembles across machines and heterogeneous resources, but
comes with several caveats:

    1. User functions registered with Globus Compute must be *non-persistent*, since
       manager-worker communicators can't be serialized or used by a remote resource.

    2. Likewise, the ``Executor.manager_poll()`` capability is disabled. The only
       available control over remote functions by workers is processing return values
       or exceptions when they complete.

    3. Globus Compute imposes a `handful of task-rate and data limits`_ on submitted functions.

    4. Users are responsible for authenticating via Globus_ and maintaining their
       `Globus Compute endpoints`_ on their target systems.

Users can still define Executor instances within their user functions and submit
MPI applications normally, as long as libEnsemble and the target application are
accessible on the remote system::

    # Within remote user function
    from libensemble.executors import MPIExecutor
    exctr = MPIExecutor()
    exctr.register_app(full_path="/home/user/forces.x", app_name="forces")
    task = exctr.submit(app_name="forces", num_procs=64)

Specify a Globus Compute endpoint in either :class:`sim_specs<libensemble.specs.SimSpecs>` or :class:`gen_specs<libensemble.specs.GenSpecs>` via the ``globus_compute_endpoint``
argument. For example::

    from libensemble.specs import SimSpecs

    sim_specs = SimSpecs(
        sim_f = sim_f,
        inputs = ["x"],
        out = [("f", float)],
        globus_compute_endpoint = "3af6dc24-3f27-4c49-8d11-e301ade15353",
    )

See the ``libensemble/tests/scaling_tests/globus_compute_forces`` directory for a complete
remote-simulation example.

Instructions for Specific Platforms
-----------------------------------

The following subsections have more information about configuring and launching
libEnsemble on specific HPC systems.

.. toctree::
    :maxdepth: 2
    :titlesonly:

    aurora
    bebop
    frontier
    improv
    perlmutter
    polaris
    summit
    srun
    example_scripts

.. _Globus Compute: https://www.globus.org/compute
.. _Globus Compute endpoints: https://globus-compute.readthedocs.io/en/latest/endpoints.html
.. _Globus: https://www.globus.org/
.. _handful of task-rate and data limits: https://globus-compute.readthedocs.io/en/latest/limits.html
