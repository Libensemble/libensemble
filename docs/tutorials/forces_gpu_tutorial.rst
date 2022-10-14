======================
Executor - Assign GPUs
======================

This tutorial shows the most portable way to assign tasks (user applications)
to the GPU.

In the first example, each worker will be using one GPU. We assume the workers are on a
cluster with CUDA-capable GPUs. We will assign GPUs by setting the environment
variable ``CUDA_VISIBLE_DEVICES``. An equivalent approach can be used with other
devices.

This example is based on the
:doc:`simple forces tutorial  <../tutorials/executor_forces_tutorial>` with
a slightly modified simulation function.

To compile the forces application to use the GPU, ensure forces.c_ has the
``#pragma omp target`` line uncommented and comment out the equivalent
``#pragma omp parallel`` line. Then compile **forces.x** using one of the
GPU build lines in build_forces.sh_ or similar for your platform.

The libEnsemble scripts in this example are available under forces_gpu_ in
the libEnsemble repository.

Note that at the time of writing, the calling script **run_libe_forces.py** is functionally
the same as that in *forces_simple*, but contains some commented out lines that can
be used for a variable resources example. The *forces_simf.py* file has slight modifications
to assign GPUs.

Videos demonstrate running this example on Perlmutter_ and Spock_.

Simulation function
-------------------

The ``sim_f`` (``forces_simf.py``) becomes as follows. The new lines are highlighted:

.. code-block:: python
    :linenos:
    :emphasize-lines: 5, 21, 23, 29-30

    import numpy as np

    # To retrieve our MPI Executor and resources instances
    from libensemble.executors.executor import Executor
    from libensemble.resources.resources import Resources

    # Optional status codes to display in libE_stats.txt for each gen or sim
    from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

    def run_forces(H, persis_info, sim_specs, libE_info):
        calc_status = 0

        # Parse out num particles, from generator function
        particles = str(int(H["x"][0][0]))

        # app arguments: num particles, timesteps, also using num particles as seed
        args = particles + " " + str(10) + " " + particles

        # Retrieve our MPI Executor instance and resources
        exctr = Executor.executor
        resources = Resources.resources.worker_resources

        resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")

        # Submit our forces app for execution. Block until the task starts.
        task = exctr.submit(
            app_name="forces",
            app_args=args,
            num_nodes=resources.local_node_count,
            procs_per_node=resources.slot_count,
            wait_on_start=True,
        )

        # Block until the task finishes
        task.wait()

        # Stat file to check for bad runs
        statfile = "forces.stat"

        # Try loading final energy reading, set the sim's status
        try:
            data = np.loadtxt(statfile)
            final_energy = data[-1]
            calc_status = WORKER_DONE
        except Exception:
            final_energy = np.nan
            calc_status = TASK_FAILED

        # Define our output array,  populate with energy reading
        outspecs = sim_specs["out"]
        output = np.zeros(1, dtype=outspecs)
        output["energy"][0] = final_energy

        # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status

The above code can be run on most systems, and will assign a GPU to each worker,
so long as the number of workers is chosen to fit the resources.

The :doc:`resource<../resource_manager/worker_resources>` attributes used are:

• **local_node_count**: The number of nodes available to this worker
• **slot_count**: The number of slots per node for this worker

and the line::

    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")

will set the environment variable ``CUDA_VISIBLE_DEVICES`` to match the assigned
slots (partitions on the node).

.. note::
    **slots** refers to the ``resource sets`` enumerated on a node (starting with
    zero). If a resource set has more than one node, then each node is considered to
    have slot zero. [:ref:`diagram<rsets-diagram>`]

Note that if you are on a system that automatically assigns free GPUs on the node,
then setting ``CUDA_VISIBLE_DEVICES`` is not necessary unless you want to ensure
workers are strictly bound to GPUs. For example, on many **SLURM** systems, you
can use ``--gpus-per-task=1`` (e.g., :doc:`Perlmutter<../platforms/perlmutter>`).
Such options can be added to the `exctr.submit` call as ``extra_args``::

    task = exctr.submit(
    ...
        extra_args='--gpus-per-task=1'
    )

Alternative environment variables can be simply substituted in ``set_env_to_slots``.
(e.g., ``HIP_VISIBLE_DEVICES``, ``ROCR_VISIBLE_DEVICES``).

.. note::
    On some systems ``CUDA_VISIBLE_DEVICES`` may be overridden by other assignments
    such as ``--gpus-per-task=1``

Running the example
-------------------

As an example, if you have been allocated two nodes, each with four GPUs, then assign
eight workers. For example::

    python run_libe_forces.py --comms local --nworkers 8

Note that if you are running one persistent generator that does not require
resources, then assign nine workers, and fix the number of *resource_sets* in
you calling script::

    libE_specs['num_resource_sets'] = 8

See :ref:`zero resource workers<zero_resource_workers>` for more ways to express this.

Changing number of GPUs per worker
----------------------------------

If you want to have two GPUs per worker on the same system (four GPUs per node),
you could assign only four workers, and change line 24 to::

    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES", multiplier=2)

In this case there are two GPUs per worker (and per slot).

Varying resources
-----------------

The same code can be used when varying worker resources. In this case, you may
add an integer field called ``resource_sets`` as a ``gen_specs['out']`` in your
calling script.

In the generator function, assign the ``resource_sets`` field of
:doc:`H<../data_structures/history_array>` for each point generated. For example
if a larger simulation requires two MPI tasks (and two GPUs), set ``resource_sets``
field to *2* for that sim_id in the generator function.

The calling script run_libe_forces.py_ contains alternative commented out lines for
a variable resource example. Search for "Uncomment for var resources"

In this case, the simulator function will still work, assigning one CPU processor
and one GPU to each MPI rank. If you want to have one rank with multiple GPUs,
then change source lines 29/30 accordingly.

Further guidance on varying resource to workers can be found under the
:doc:`resource manager<../resource_manager/resources_index>`.

Checking GPU usage
------------------

You can check you are running forces on the GPUs as expected by using profiling tools and/or by using
a monitoring utility. For NVIDIA GPUs, for example, the **Nsight** profiler is generally available
and can be run from the command line. To simply run `forces.x` stand-alone you could run::

    nsys profile --stats=true mpirun -n 2 ./forces.x

To use the `nvidia-smi` monitoring tool while running, open another shell where your code is
running (this may entail using *ssh* to get on to the node), and run::

    watch -n 0.1 nvidia-smi

This will update GPU usage information every 0.1 seconds. You would need to ensure the code
runs for long enough to register on the monitor, so lets try 100,000 particles::

    mpirun -n 2 ./forces.x 100000

It is also recommended that you run without the profiler when using the `nvidia-smi` utility.

This can also be used when running via libEnsemble, so long as you are on the node where the
forces applications are being run. As the default particles in the forces example is 1000, you
will need to to increase particles to see clear GPU usage in the live monitor. E.g.,~ in line 14
to multiply the particles by 10::

        # Parse out num particles, from generator function
        particles = str(int(H["x"][0][0]) * 10)

Alternative monitoring devices include ``rocm-smi`` (AMD) and ``intel_gpu_top`` (Intel). The latter
does not need the *watch* command.

Example submission script
-------------------------

A simple example batch script for :doc:`Perlmutter<../platforms/perlmutter>`
that runs 8 workers on 2 nodes:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J libE_small_test
    #SBATCH -A <myproject_g>
    #SBATCH -C gpu
    #SBATCH --time 10
    #SBATCH --nodes 2

    export MPICH_GPU_SUPPORT_ENABLED=1
    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

    python run_libe_forces.py --comms local --nworkers 8

where ``SLURM_EXACT`` and ``SLURM_MEM_PER_NODE`` are set to prevent
resource conflicts on each node.

.. _forces_gpu: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_gpu
.. _forces.c: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_app/forces.c
.. _build_forces.sh: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_app/build_forces.sh
.. _Perlmutter: https://www.youtube.com/watch?v=Av8ctYph7-Y
.. _Spock: https://www.youtube.com/watch?v=XHXcslDORjU
.. _run_libe_forces.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_gpu/run_libe_forces.py
