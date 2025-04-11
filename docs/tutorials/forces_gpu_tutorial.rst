======================
Executor - Assign GPUs
======================

This tutorial shows the most portable way to assign tasks (user applications)
to the GPU. The libEnsemble scripts in this example are available under
forces_gpu_ in the libEnsemble repository.

This example is based on the
:doc:`simple forces tutorial <../tutorials/executor_forces_tutorial>` with
a slightly modified simulation function (to assign GPUs) and a greatly increased
number of particles (to allow real-time GPU usage to be viewed).

In the first example, each worker will be using one GPU. The code will assign the
GPUs available to each worker, using the appropriate method. This works on systems
using **Nvidia**, **AMD**, and **Intel** GPUs without modifying the scripts.

A video demonstrates running this example on Frontier_.

Simulation function
-------------------

The ``sim_f`` (``forces_simf.py``) is as follows. The lines that are different
from the simple forces example are highlighted:

.. code-block:: python
    :linenos:
    :emphasize-lines: 31-32, 39

    import numpy as np

    # Optional status codes to display in libE_stats.txt for each gen or sim
    from libensemble.message_numbers import TASK_FAILED, WORKER_DONE

    # Optional - to print GPU settings
    from libensemble.tools.test_support import check_gpu_setting

    def run_forces(H, persis_info, sim_specs, libE_info):
        """Launches the forces MPI app and auto-assigns ranks and GPU resources.

        Assigns one MPI rank to each GPU assigned to the worker.
        """

        calc_status = 0

        # Parse out num particles, from generator function
        particles = str(int(H["x"][0][0]))

        # app arguments: num particles, timesteps, also using num particles as seed
        args = particles + " " + str(10) + " " + particles

        # Retrieve our MPI Executor
        exctr = libE_info["executor"]

        # Submit our forces app for execution.
        task = exctr.submit(
            app_name="forces",
            app_args=args,
            auto_assign_gpus=True,
            match_procs_to_gpus=True,
        )

        # Block until the task finishes
        task.wait()

        # Optional - prints GPU assignment (method and numbers)
        check_gpu_setting(task, assert_setting=False, print_setting=True)

        # Try loading final energy reading, set the sim's status
        statfile = "forces.stat"
        try:
            data = np.loadtxt(statfile)
            final_energy = data[-1]
            calc_status = WORKER_DONE
        except Exception:
            final_energy = np.nan
            calc_status = TASK_FAILED

        # Define our output array, populate with energy reading
        output = np.zeros(1, dtype=sim_specs["out"])
        output["energy"] = final_energy

        # Return final information to worker, for reporting to manager
        return output, persis_info, calc_status

Lines 31-32 tell the executor to use the GPUs assigned to this worker, and
to match processors (MPI ranks) to GPUs.

The user can also set ``num_procs`` and ``num_gpus`` in the generator as in
the `forces_gpu_var_resources`_ example, and skip lines 31-32.

Line 37 simply prints out how the GPUs were assigned. If this is not as expected,
:ref:`platform configuration<datastruct-platform-specs>` can be provided.

While this is sufficient for most users, note that it is possible to query
the resources assigned to *this* worker (nodes and partitions of nodes),
and use this information however you want.

.. dropdown:: How to query this worker's resources

    The example shown below implements
    a similar, but less portable, version of the above (excluding output lines).

    .. code-block:: python
        :linenos:
        :emphasize-lines: 5, 22, 24, 30-31

        import numpy as np

        # To retrieve our MPI Executor and resources instances
        from libensemble.executors.executor import Executor
        from libensemble.resources.resources import Resources

        # Optional status codes to display in libE_stats.txt for each gen or sim
        from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


        def run_forces(H, _, sim_specs):
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

            # Read final energy
            data = np.loadtxt(statfile)
            final_energy = data[-1]

            # Define our output array,  populate with energy reading
            output = np.zeros(1, dtype=sim_specs["out"])
            output["energy"][0] = final_energy

        return output

    The above code will assign a GPU to each worker on CUDA-capable systems,
    so long as the number of workers is chosen to fit the resources.

    If you want to have one rank with multiple GPUs, then change source lines 30/31
    accordingly.

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
            extra_args="--gpus-per-task=1"
        )

    Alternative environment variables can be simply substituted in ``set_env_to_slots``.
    (e.g., ``HIP_VISIBLE_DEVICES``, ``ROCR_VISIBLE_DEVICES``).

    .. note::
        On some systems ``CUDA_VISIBLE_DEVICES`` may be overridden by other assignments
        such as ``--gpus-per-task=1``

Compiling the Forces application
--------------------------------

First, compile the forces application under the ``forces_app`` directory.

Compile **forces.x** using one of the GPU build lines in build_forces.sh_
or similar for your platform.

Running the example
-------------------

As an example, if you have been allocated two nodes, each with four GPUs, then assign
nine workers (the extra worker runs the persistent generator).

For example::

    python run_libe_forces.py --nworkers 9

See :ref:`zero-resource workers<zero_resource_workers>` for more ways to express this.

Changing the number of GPUs per worker
--------------------------------------

If you want to have two GPUs per worker on the same system (with four GPUs per node),
you could assign only four workers. You will see that two GPUs are used for each
forces run.

.. _var-resources-gpu:

Varying resources
-----------------

A variant of this example where you may specify any number of processors
and GPUs for each simulation is given in the `forces_gpu_var_resources`_ example.

In this example, when simulations are parameterized in the generator function,
the ``gen_specs["out"]`` field ``num_gpus`` is set for each simulation (based
on the number of particles). These values will automatically be used for each
simulation (they do not need to be passed as a ``sim_specs["in"]``).

Further guidance on varying the resources assigned to workers can be found under the
:doc:`resource manager<../resource_manager/resources_index>` section.

Multiple applications
---------------------

Another variant of this example, forces_multi_app_, has two applications, one that
uses GPUs, and another that only uses CPUs. Dynamic resource management can
manage both types of resources and assign these to the same nodes concurrently, for
maximum efficiency.

Checking GPU usage
------------------

The output of `forces.x` will say if it has run on the host or device. When running
libEnsemble, this can be found in the simulation directories (under the ``ensemble``
directory).

You can check you are running forces on the GPUs as expected by using profiling tools and/or
by using a monitoring utility. For NVIDIA GPUs, for example, the **Nsight** profiler is
generally available and can be run from the command line. To simply run `forces.x` stand-alone
you could run::

    nsys profile --stats=true mpirun -n 2 ./forces.x

To use the `nvidia-smi` monitoring tool while running, open another shell where your code is
running (this may entail using *ssh* to get on to the node), and run::

    watch -n 0.1 nvidia-smi

This will update GPU usage information every 0.1 seconds. You would need to ensure the code
runs for long enough to register on the monitor, so let's try 100,000 particles::

    mpirun -n 2 ./forces.x 100000

It is also recommended that you run without the profiler when using the `nvidia-smi` utility.

This can also be used when running via libEnsemble, so long as you are on the node where the
forces applications are being run.

Alternative monitoring devices include ``rocm-smi`` (AMD) and ``intel_gpu_top`` (Intel).
The latter does not need the *watch* command.

Example submission script
-------------------------

A simple example batch script for :doc:`Perlmutter<../platforms/perlmutter>`
that runs 8 workers on 2 nodes:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J libE_small_test
    #SBATCH -A <myproject>
    #SBATCH -C gpu
    #SBATCH --time 10
    #SBATCH --nodes 2

    export MPICH_GPU_SUPPORT_ENABLED=1
    export SLURM_EXACT=1

    python run_libe_forces.py --nworkers 9

where ``SLURM_EXACT`` is set to help prevent resource conflicts on each node.

.. _build_forces.sh: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_app/build_forces.sh
.. _forces.c: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_app/forces.c
.. _forces_gpu: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_gpu
.. _forces_gpu_var_resources: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_gpu_var_resources/run_libe_forces.py
.. _forces_multi_app: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_multi_app/run_libe_forces.py
.. _Frontier: https://youtu.be/H2fmbZ6DnVc
.. _Perlmutter: https://www.youtube.com/watch?v=Av8ctYph7-Y
.. _Polaris: https://youtu.be/Ff0dYYLQzoU
