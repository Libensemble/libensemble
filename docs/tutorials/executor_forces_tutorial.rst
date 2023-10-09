==================================
Executor with Electrostatic Forces
==================================

This tutorial highlights libEnsemble's capability to portably execute
and monitor external scripts or user applications within simulation or generator
functions using the :doc:`executor<../executor/overview>`.

This tutorial's calling script registers a compiled executable that simulates
electrostatic forces between a collection of particles. The simulator function
launches instances of this executable and reads output files to determine
the result.

This tutorial uses libEnsemble's :doc:`MPI Executor<../executor/mpi_executor>`,
which automatically detects available MPI runners and resources.

This example also uses a persistent generator. This generator runs on a
worker throughout the ensemble, producing new simulation parameters as requested.

Getting Started
---------------

The simulation source code ``forces.c`` can be obtained directly from the
libEnsemble repository in the forces_app_ directory.

Assuming MPI and its C compiler ``mpicc`` are available, compile
``forces.c`` into an executable (``forces.x``) with:

.. code-block:: bash

    $ mpicc -O3 -o forces.x forces.c -lm

Alternative build lines for different platforms can be found in the ``build_forces.sh``
file in the same directory.

Calling Script
--------------

Complete scripts for this example can be found in the forces_simple_ directory.

Let's begin by writing our calling script to specify our simulation and
generation functions and call libEnsemble. Create a Python file called
`run_libe_forces.py` containing:

.. code-block:: python
    :linenos:
    :emphasize-lines: 16,24,27

    #!/usr/bin/env python
    import os
    import sys

    import numpy as np
    from forces_simf import run_forces  # Sim func from current dir

    from libensemble import Ensemble
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
    from libensemble.executors import MPIExecutor
    from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
    from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    if __name__ == "__main__":
        # Initialize MPI Executor
        exctr = MPIExecutor()

        # Register simulation executable with executor
        sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

        if not os.path.isfile(sim_app):
            sys.exit("forces.x not found - please build first in ../forces_app dir")

        exctr.register_app(full_path=sim_app, app_name="forces")

        # Parse number of workers, comms type, etc. from arguments
        ensemble = Ensemble(parse_args=True, executor=exctr)

On line 16, we instantiate our :doc:`MPI Executor<../executor/mpi_executor>`.

Registering an application is as easy as providing the full file-path and giving
it a memorable name. This Executor will later be used within our simulation
function to launch the registered app.

On line 27, we initialize the ensemble. The :meth:`parse_args<tools.parse_args>`
parameter is used to read `comms` and `nworkers` from the command line. This sets
the respective `libE_specs` options.

Next, we add basic configuration for the ensemble. As one worker will run a persistent
generator, we calculate the number of workers that need resources to run simulations.
We also set `sim_dirs_make` so that a directory is created for each simulation. This
helps organize output and also helps prevent workers from overwriting previous results.

.. code-block:: python
  :linenos:
  :lineno-start: 30

    nsim_workers = ensemble.nworkers - 1  # One worker is for persistent generator

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        sim_dirs_make=True,
    )

Next we define the :ref:`sim_specs<datastruct-sim-specs>` and
:ref:`gen_specs<datastruct-gen-specs>`. Recall that these are used to specify
to libEnsemble what user functions and input/output fields to
expect, and also to parameterize user functions:

.. code-block:: python
  :linenos:
  :lineno-start: 38

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        inputs=["x"],
        outputs=[("energy", float)],
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when starting persistent generator
        persis_in=["sim_id"],  # Return sim_ids of evaluated points to generator
        outputs=[("x", float, (1,))],
        user={
            "initial_batch_size": nsim_workers,
            "lb": np.array([1000]),  # min particles
            "ub": np.array([3000]),  # max particles
        },
    )

Next, configure an allocation function, which starts the one persistent
generator and farms out the simulations. We also tell it to wait for all
simulations to return their results, before generating more parameters.

.. code-block:: python
  :linenos:
  :lineno-start: 56

    # Starts one persistent generator. Simulated values are returned in batch.
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": False,  # False causes batch returns
        },
    )

Now we set :ref:`exit_criteria<datastruct-exit-criteria>` to
exit after running eight simulations.

We also give each worker a seeded random stream, via the
:ref:`persis_info<datastruct-persis-info>`  option.
These can be used for random number generation if required.

Finally we :doc:`run<../libe_module>` the ensemble.

.. code-block:: python
  :linenos:
  :lineno-start: 64

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=8)

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

Exercise
^^^^^^^^

This may take some additional browsing of the docs to complete.

Write an alternative Calling Script similar to above, but with the following differences:

 1. Set :ref:`libEnsemble's logger<logger_config>` to print debug messages.
 2. Override the MPIExecutor's detected MPI runner with ``"openmpi"``.
 3. Tell the allocation function to return results to the generator asynchronously.
 4. Use the ensemble function :meth:`save_output()<libensemble.ensemble.Ensemble.save_output>` to save the History array and ``persis_info`` to files after libEnsemble completes.

.. dropdown:: **Click Here for Solutions**

   **Soln 1.** Debug logging gives lots of information.

   .. code-block:: python
       :linenos:
       :lineno-start: 13

       from libensemble import logger

       logger.set_level("DEBUG")

   **Soln 2.** This can also be specified via :ref:`platform_specs<datastruct-platform-specs>` option.

   .. code-block:: python
       :linenos:
       :lineno-start: 16

        # Initialize MPI Executor
        exctr = MPIExecutor(custom_info={"mpi_runner": "openmpi"})

   **Soln 3.** Set ``async_return`` to *True*.

   .. code-block:: python
       :linenos:
       :lineno-start: 56
       :emphasize-lines: 5

        # Starts one persistent generator. Simulated values are returned in batch.
        ensemble.alloc_specs = AllocSpecs(
            alloc_f=alloc_f,
            user={
                "async_return": True,
            },
        )

   **Soln 4.** This will save the output based on the name of the calling script. You
   can give any string in place of ``__file__``.

   .. code-block:: python
       :linenos:
       :lineno-start: 72

       ensemble.save_output(__file__)

Simulation Function
-------------------

Our simulation function is where we'll use libEnsemble's executor to configure and submit
our application for execution. We'll poll this task's state while
it runs, and once we've detected it has finished we'll send any results or
exit statuses back to the manager.

Create another Python file named ``forces_simf.py`` containing the following
for starters:

.. code-block:: python
    :linenos:

    import numpy as np

    # Optional status codes to display in libE_stats.txt for each gen or sim
    from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


    def run_forces(H, persis_info, sim_specs, libE_info):
        """Runs the forces MPI application"""

        calc_status = 0

        # Parse out num particles, from generator function
        particles = str(int(H["x"][0][0]))

        # app arguments: num particles, timesteps, also using num particles as seed
        args = particles + " " + str(10) + " " + particles

        # Retrieve our MPI Executor
        exctr = libE_info["executor"]

        # Submit our forces app for execution.
        task = exctr.submit(app_name="forces", app_args=args)

        # Block until the task finishes
        task.wait()

We retrieve the generated number of particles from ``H`` and construct
an argument string for our launched application. The particle count doubles up
as a random number seed here.

We then retrieve our previously instantiated Executor. libEnsemble will use
the MPI runner detected (or provided by :ref:`platform options<datastruct-platform-specs>`).
As `num_procs` (or similar) is not specified, libEnsemble will assign the processors
available to this worker.

After submitting the "forces" app for execution,
a :ref:`Task<task_tag>` object is returned that correlates with the launched app.
This object is roughly equivalent to a Python future and can be polled, killed,
and evaluated in a variety of helpful ways. For now, we're satisfied with waiting
for the task to complete via ``task.wait()``.

We can assume that afterward, any results are now available to parse. Our application
produces a ``forces.stat`` file that contains either energy
computations for every timestep or a "kill" message if particles were lost, which
indicates a bad run - this can be ignored for now.

To complete our simulation function, parse the last energy value from the output file into
a local output :ref:`History array<funcguides-history>`, and if successful,
set the simulation function's exit status :ref:`calc_status<funcguides-calcstatus>`
to ``WORKER_DONE``. Otherwise, send back ``NAN`` and a ``TASK_FAILED`` status:

.. code-block:: python
    :linenos:
    :lineno-start: 27

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

``calc_status`` will be displayed in the ``libE_stats.txt`` log file.

That's it! As can be seen, with libEnsemble, it's relatively easy to get started
with launching applications.

Running the example
-------------------

This completes our calling script and simulation function. Run libEnsemble with:

.. code-block:: bash

    $ python run_libe_forces.py --comms local --nworkers [nworkers]

where ``nworkers`` is one more than the number of concurrent simulations.

Output files (including ``forces.stat`` and files containing ``stdout`` and
``stderr`` content for each task) should appear in the current working
directory. Overall workflow information should appear in ``libE_stats.txt``
and ``ensemble.log`` as usual.

.. dropdown:: **Example run / output**

   For example, after running:

   .. code-block:: bash

       $ python run_libe_forces.py --comms local --nworkers 3

   my ``libE_stats.txt`` resembled::

     Manager     : Starting ensemble at: 2023-09-12 18:12:08.517
     Worker     2: sim_id     0: sim Time: 0.205 Start: ... End: ... Status: Completed
     Worker     3: sim_id     1: sim Time: 0.284 Start: ... End: ... Status: Completed
     Worker     2: sim_id     2: sim Time: 0.117 Start: ... End: ... Status: Completed
     Worker     3: sim_id     3: sim Time: 0.294 Start: ... End: ... Status: Completed
     Worker     2: sim_id     4: sim Time: 0.124 Start: ... End: ... Status: Completed
     Worker     3: sim_id     5: sim Time: 0.174 Start: ... End: ... Status: Completed
     Worker     3: sim_id     7: sim Time: 0.135 Start: ... End: ... Status: Completed
     Worker     2: sim_id     6: sim Time: 0.275 Start: ... End: ... Status: Completed
     Worker     1: Gen no     1: gen Time: 1.038 Start: ... End: ... Status: Persis gen finished
     Manager     : Exiting ensemble at: 2023-09-12 18:12:09.565 Time Taken: 1.048

   where ``status`` is set based on the simulation function's returned ``calc_status``.

   My ``ensemble.log`` (on a four-core laptop) resembled::

     [0]  ... libensemble.libE (INFO): Logger initializing: [workerID] precedes each line. [0] = Manager
     [0]  ... libensemble.libE (INFO): libE version v0.10.2+dev
     [0]  ... libensemble.manager (INFO): Manager initiated on node shuds
     [0]  ... libensemble.manager (INFO): Manager exit_criteria: {'sim_max': 8}
     [2]  ... libensemble.worker (INFO): Worker 2 initiated on node shuds
     [3]  ... libensemble.worker (INFO): Worker 3 initiated on node shuds
     [1]  ... libensemble.worker (INFO): Worker 1 initiated on node shuds
     [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_0: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 2023 10 2023
     [3]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker3_0: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 2900 10 2900
     [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_0 finished with errcode 0 (FINISHED)
     [3]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker3_0 finished with errcode 0 (FINISHED)
     [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_1: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 1288 10 1288
     [3]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker3_1: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 2897 10 2897
     [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_1 finished with errcode 0 (FINISHED)
     [3]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker3_1 finished with errcode 0 (FINISHED)
     [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_2: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 1623 10 1623
     [3]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker3_2: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 1846 10 1846
     [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_2 finished with errcode 0 (FINISHED)
     [3]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker3_2 finished with errcode 0 (FINISHED)
     [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_3: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 2655 10 2655
     [3]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker3_3: mpirun -hosts shuds -np 2 --ppn 2 /home/.../forces_app/forces.x 1818 10 1818
     [3]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker3_3 finished with errcode 0 (FINISHED)
     [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_3 finished with errcode 0 (FINISHED)
     [0]  ... libensemble.manager (INFO): Term test tripped: sim_max
     [0]  ... libensemble.manager (INFO): Term test tripped: sim_max
     [0]  ... libensemble.libE (INFO): Manager total time: 1.043

   Note again that the four cores were divided equally among two workers that run simulations.

That concludes this tutorial. Each of these example files can be found in the
repository in `examples/tutorials/forces_with_executor`_.

For further experimentation, we recommend trying out this libEnsemble tutorial
workflow on a cluster or multi-node system, since libEnsemble can also manage
those resources and is developed to coordinate computations at huge scales.
See :ref:`HPC platform guides<platform-index>` for more information.

See the :doc:`forces_gpu tutorial<forces_gpu_tutorial>` for a similar workflow
including GPUs. That tutorial also shows how to dynamically assign resources to
each simulation.

Please feel free to contact us or open an issue on GitHub_ if this tutorial
workflow doesn't work properly on your cluster or other compute resource.

Exercises
^^^^^^^^^

These may require additional browsing of the documentation to complete.

  1. Adjust :meth:`submit()<libensemble.executors.mpi_executor.MPIExecutor.submit>` to launch with four processes.
  2. Adjust ``submit()`` again so the app's ``stdout`` and ``stderr`` are written to ``stdout.txt`` and ``stderr.txt`` respectively.
  3. Add a fourth argument to the args line to make 20% of simulations go bad.
  4. Construct a ``while not task.finished:`` loop that periodically sleeps for a tenth of a second, calls :meth:`task.poll()<libensemble.executors.executor.Task.poll>`,
     then reads the output ``.stat`` file, and calls :meth:`task.kill()<libensemble.executors.executor.Task.kill>` if the output file contains ``"kill\n"``
     or if ``task.runtime`` exceeds sixty seconds.

.. dropdown:: **Click Here for Solution**

   Showing updated sections only (``---`` refers to snips where code is unchanged).

   .. code-block:: python

        import time

        ...
        args = particles + " " + str(10) + " " + particles + " " + str(0.2)
        ...
        statfile = "forces.stat"
        task = exctr.submit(
            app_name="forces",
            app_args=args,
            num_procs=4,
            stdout="stdout.txt",
            stderr="stderr.txt",
        )

        while not task.finished:
            time.sleep(0.1)
            task.poll()

            if task.file_exists_in_workdir(statfile):
                with open(statfile, "r") as f:
                    if "kill\n" in f.readlines():
                        task.kill()

            if task.runtime > 60:
                task.kill()

        ...

.. _examples/tutorials/forces_with_executor: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/forces_with_executor
.. _forces_app: https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/forces/forces_app
.. _forces_simple: https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/forces/forces_simple
.. _GitHub: https://github.com/Libensemble/libensemble/issues
