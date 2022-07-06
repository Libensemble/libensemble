==================================
Executor with Electrostatic Forces
==================================

This tutorial highlights libEnsemble's capability to execute
and monitor external scripts or user applications within simulation or generator
functions using the :doc:`executor<../executor/overview>`. In this tutorial,
our calling script registers a compiled executable that simulates
electrostatic forces between a collection of particles. The simulator function
launches instances of this executable and reads output files to determine
if the run was successful.

It is possible to use ``subprocess`` calls from Python to issue
commands such as ``jsrun`` or ``aprun`` to run applications. Unfortunately,
hard-coding such commands within user scripts isn't portable.
Furthermore, many systems like Argonne's :doc:`Theta<../platforms/theta>` do not
allow libEnsemble to submit additional tasks from the compute nodes. On these
systems, a proxy launch mechanism (such as Balsam) is required.
libEnsemble's Executors were developed to directly address such issues.

In particular, we'll be experimenting with
libEnsemble's :doc:`MPI Executor<../executor/mpi_executor>`, since it can automatically
detect available MPI runners and resources, and by default divide them equally among workers.

Getting Started
---------------

The simulation source code ``forces.c`` can be obtained directly from the
libEnsemble repository here_.

Assuming MPI and its C compiler ``mpicc`` are available, compile
``forces.c`` into an executable (``forces.x``) with:

.. code-block:: bash

    $ mpicc -O3 -o forces.x forces.c -lm

Calling Script
--------------

Let's begin by writing our calling script to parameterize our simulation and
generation functions and call libEnsemble. Create a Python file called `run_libe_forces.py`
containing:

.. code-block:: python
    :linenos:
    :emphasize-lines: 15,19

    #!/usr/bin/env python
    import os
    import numpy as np
    from forces_simf import run_forces  # Sim func from current dir

    from libensemble.libE import libE
    from libensemble.gen_funcs.sampling import uniform_random_sample
    from libensemble.tools import parse_args, add_unique_random_streams
    from libensemble.executors import MPIExecutor

    # Parse number of workers, comms type, etc. from arguments
    nworkers, is_manager, libE_specs, _ = parse_args()

    # Initialize MPI Executor instance
    exctr = MPIExecutor()

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")
    exctr.register_app(full_path=sim_app, app_name="forces")

On line 15, we instantiate our :doc:`MPI Executor<../executor/mpi_executor>` class instance,
which can optionally be customized by specifying alternative MPI runners. The
auto-detected default should be sufficient.

Registering an application is as easy as providing the full file-path and giving
it a memorable name. This Executor instance will later be retrieved within our
simulation function to launch the registered app.

Next define the :ref:`sim_specs<datastruct-sim-specs>` and
:ref:`gen_specs<datastruct-gen-specs>` data structures. Recall that these
are used to specify to libEnsemble what user functions and input/output fields to
expect, and also to parameterize function instances without hard-coding:

.. code-block:: python
    :linenos:

    # State the sim_f, inputs, outputs
    sim_specs = {
        "sim_f": run_forces,  # sim_f, imported above
        "in": ["x"],  # Name of input for sim_f
        "out": [("energy", float)],  # Name, type of output from sim_f
    }

    # State the gen_f, inputs, outputs, additional parameters
    gen_specs = {
        "gen_f": uniform_random_sample,  # Generator function
        "in": [],  # Generator input
        "out": [("x", float, (1,))],  # Name, type, and size of data from gen_f
        "user": {
            "lb": np.array([1000]),  # User parameters for the gen_f
            "ub": np.array([3000]),
            "gen_batch_size": 8,
        },
    }

Our generation function will generate random numbers of particles (between
the ``"lb"`` and ``"ub"`` bounds) for our simulation function to evaluate via our
registered application.

The following additional :ref:`libE_specs setting<output_dirs>` instructs libEnsemble's workers
to each create and work within a separate directory each time they call a simulation
function. This helps organize output and also helps prevents workers from overwriting
previous results:

.. code-block:: python
    :linenos:

    # Create and work inside separate per-simulation directories
    libE_specs['sim_dirs_make'] = True

After configuring :ref:`persis_info<datastruct-persis-info>` and
:ref:`exit_criteria<datastruct-exit-criteria>`, we initialize libEnsemble
by calling the primary :doc:`libE()<../libe_module>` routine:

.. code-block:: python
  :linenos:

  # Instruct libEnsemble to exit after this many simulations
  exit_criteria = {"sim_max": 8}

  # Seed random streams for each worker, particularly for gen_f
  persis_info = add_unique_random_streams({}, nworkers + 1)

  # Launch libEnsemble
  H, persis_info, flag = libE(
      sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs
  )

Exercise
^^^^^^^^

This may take some additional browsing of the docs to complete.

Write an alternative Calling Script similar to above, but with the following differences:

 1. Add an additional :ref:`worker directory setting<output_dirs>` so workers operate in ``/scratch/ensemble`` instead of the default current working directory.
 2. Override the MPIExecutor's detected MPI runner with ``'openmpi'``.
 3. Set :ref:`libEnsemble's logger<logger_config>` to print debug messages.
 4. Use the :meth:`save_libE_output()<tools.save_libE_output>` function to save the History array and ``persis_info`` to files after libEnsemble completes.

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

        #!/usr/bin/env python
        import os
        import numpy as np
        from forces_simf import run_forces  # Sim func from current dir

        from libensemble import logger
        from libensemble.libE import libE
        from libensemble.gen_funcs.sampling import uniform_random_sample
        from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
        from libensemble.executors import MPIExecutor

        # Parse number of workers, comms type, etc. from arguments
        nworkers, is_manager, libE_specs, _ = parse_args()

        # Adjust logger level
        logger.set_level('DEBUG')

        # Initialize MPI Executor instance
        exctr = MPIExecutor(custom_info={'mpi_runner': 'openmpi'})

        ...

        # Instruct workers to operate somewhere else on the filesystem
        libE_specs['ensemble_dir_path'] = "/scratch/ensemble"

        ...

        # Launch libEnsemble
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs
        )

        if is_manager:
            save_libE_output(H, persis_info, __file__, nworkers)

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

    # To retrieve our MPI Executor instance
    from libensemble.executors.executor import Executor

    # Optional status codes to display in libE_stats.txt for each gen or sim
    from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

    def run_forces(H, persis_info, sim_specs, libE_info):
        calc_status = 0

        # Parse out num particles, from generator function
        particles = str(int(H["x"][0][0]))

        # num particles, timesteps, also using num particles as seed
        args = particles + " " + str(10) + " " + particles

        # Retrieve our MPI Executor instance
        exctr = Executor.executor

        # Submit our forces app for execution
        task = exctr.submit(app_name="forces", app_args=args)

        # Block until the task finishes
        task.wait()

We retrieve the generated number of particles from ``H`` and construct
an argument string for our launched application. The particle count doubles up
as a random number seed here. Note a fourth argument can be added to forces
that gives forces a chance of a 'bad run' (a float between 0 and 1), but
for now that will default to zero.

We then retrieve our previously instantiated Executor instance from the
class definition, where it was automatically stored as an attribute.

After submitting the "forces" app for execution,
a :ref:`Task<task_tag>` object is returned that correlates with the launched app.
This object is roughly equivalent to a Python future, and can be polled, killed,
and evaluated in a variety of helpful ways. For now, we're satisfied with waiting
for the task to complete via ``task.wait()``.

We can assume that afterward, any results are now available to parse. Our application
produces a ``forces.stat`` file that contains either energy
computations for every time-step or a "kill" message if particles were lost, which
indicates a bad run - this can be ignored for now.

To complete our simulation function, parse the last energy value from the output file into
a local output :ref:`History array<datastruct-history-array>`, and if successful,
set the simulation function's exit status :ref:`calc_status<datastruct-calc-status>`
to ``WORKER_DONE``. Otherwise, send back ``NAN`` and a ``TASK_FAILED`` status:

.. code-block:: python
    :linenos:

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

``calc_status`` will be displayed in the ``libE_stats.txt`` log file.

That's it! As can be seen, with libEnsemble, it's relatively easy to get started
with launching applications. Behind the scenes, libEnsemble evaluates default
MPI runners and available resources and divides them among the workers.

This completes our calling script and simulation function. Run libEnsemble with:

.. code-block:: bash

    $ python run_libe_forces.py --comms local --nworkers [nworkers]

This may take up to a minute to complete. Output files---including ``forces.stat``
and files containing ``stdout`` and ``stderr`` content for each task---should
appear in the current working directory. Overall workflow information
should appear in ``libE_stats.txt`` and ``ensemble.log`` as usual.

For example, my ``libE_stats.txt`` resembled::

  Worker     1: Gen no     1: gen Time: 0.001 Start: ... End: ... Status: Not set
  Worker     1: sim_id     0: sim Time: 0.227 Start: ... End: ... Status: Completed
  Worker     2: sim_id     1: sim Time: 0.426 Start: ... End: ... Status: Completed
  Worker     1: sim_id     2: sim Time: 0.627 Start: ... End: ... Status: Completed
  Worker     2: sim_id     3: sim Time: 0.225 Start: ... End: ... Status: Completed
  Worker     1: sim_id     4: sim Time: 0.224 Start: ... End: ... Status: Completed
  Worker     2: sim_id     5: sim Time: 0.625 Start: ... End: ... Status: Completed
  Worker     1: sim_id     6: sim Time: 0.225 Start: ... End: ... Status: Completed
  Worker     2: sim_id     7: sim Time: 0.626 Start: ... End: ... Status: Completed

Where ``status`` is set based on the simulation function's returned ``calc_status``.

My ``ensemble.log`` (on a ten-core laptop) resembled::

  [0]  ... libensemble.libE (INFO): Logger initializing: [workerID] precedes each line. [0] = Manager
  [0]  ... libensemble.libE (INFO): libE version v0.9.0
  [0]  ... libensemble.manager (INFO): Manager initiated on node my_laptop
  [0]  ... libensemble.manager (INFO): Manager exit_criteria: {'sim_max': 8}
  [1]  ... libensemble.worker (INFO): Worker 1 initiated on node my_laptop
  [2]  ... libensemble.worker (INFO): Worker 2 initiated on node my_laptop
  [1]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker1_0: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 2023 10 2023
  [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_0: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 2900 10 2900
  [1]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker1_0 finished with errcode 0 (FINISHED)
  [1]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker1_1: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 1288 10 1288
  [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_0 finished with errcode 0 (FINISHED)
  [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_1: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 2897 10 2897
  [1]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker1_1 finished with errcode 0 (FINISHED)
  [1]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker1_2: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 1623 10 1623
  [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_1 finished with errcode 0 (FINISHED)
  [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_2: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 1846 10 1846
  [1]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker1_2 finished with errcode 0 (FINISHED)
  [1]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker1_3: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 2655 10 2655
  [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_2 finished with errcode 0 (FINISHED)
  [2]  ... libensemble.executors.mpi_executor (INFO): Launching task libe_task_forces_worker2_3: mpirun -hosts my_laptop -np 5 --ppn 5 /Users/.../forces.x 1818 10 1818
  [1]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker1_3 finished with errcode 0 (FINISHED)
  [2]  ... libensemble.executors.executor (INFO): Task libe_task_forces_worker2_3 finished with errcode 0 (FINISHED)
  [0]  ... libensemble.manager (INFO): Term test tripped: sim_max
  [0]  ... libensemble.manager (INFO): Term test tripped: sim_max
  [0]  ... libensemble.libE (INFO): Manager total time: 3.939

Note again that the ten cores were divided equally among two workers.

That concludes this tutorial.
Each of these example files can be found in the repository in `examples/tutorials/forces_with_executor`_.

For further experimentation, we recommend trying out this libEnsemble tutorial
workflow on a cluster or multi-node system, since libEnsemble can also manage
those resources and is developed to coordinate computations at huge scales.
Please feel free to contact us or open an issue on GitHub_ if this tutorial
workflow doesn't work properly on your cluster or other compute resource.

Exercises
^^^^^^^^^

These may require additional browsing of the documentation to complete.

  1. Adjust :meth:`submit()<mpi_executor.MPIExecutor.submit>` to launch with four processes.
  2. Adjust ``submit()`` again so the app's ``stdout`` and ``stderr`` are written to ``stdout.txt`` and ``stderr.txt`` respectively.
  3. Add a fourth argument to the args line to make 20% of simulations go bad.
  4. Construct a ``while not task.finished:`` loop that periodically sleeps for a tenth of a second, calls :meth:`task.poll()<executor.Task.poll>`,
     then reads the output ``.stat`` file, and calls :meth:`task.kill()<executor.Task.kill>` if the output file contains ``"kill\n"``
     or if ``task.runtime`` exceeds sixty seconds.

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

        import time
        ...
        args = particles + " " + str(10) + " " + particles + " " + str(0.2)
        ...
        statfile = "forces.stat"
        task = exctr.submit(app_name="forces", app_args=args,
                            num_procs=4,
                            stdout="stdout.txt",
                            stderr="stderr.txt")

        while not task.finished:
          time.sleep(0.1)
          task.poll()

          if task.file_exists_in_workdir(statfile):
            with open(statfile, 'r') as f:
                if "kill\n" in f.readlines():
                    task.kill()

          if task.runtime > 60:
            task.kill()

        ...

.. _here: https://raw.githubusercontent.com/Libensemble/libensemble/main/libensemble/tests/scaling_tests/forces/forces.c
.. _examples/tutorials/forces_with_executor: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/forces_with_executor
.. _GitHub: https://github.com/Libensemble/libensemble/issues
