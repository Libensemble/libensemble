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
systems a proxy launch mechanism (such as Balsam) is required.
libEnsemble's various Executors were developed to directly address such issues.

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
generation functions and call libEnsemble. Create a Python file containing:

.. code-block:: python
    :linenos:
    :emphasize-lines: 15,19

    #!/usr/bin/env python
    import os
    import numpy as np
    from tutorial_forces_simf_simple import run_forces  # Sim func from current dir

    from libensemble.libE import libE
    from libensemble.gen_funcs.sampling import uniform_random_sample
    from libensemble.tools import parse_args, add_unique_random_streams
    from libensemble.executors import MPIExecutor

    # Parse number of workers, comms type, etc. from arguments
    nworkers, is_manager, libE_specs, _ = parse_args()

    # Initialize MPI Executor instance
    exctr = MPIExecutor()

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), "forces.x")
    exctr.register_app(full_path=sim_app, app_name="forces")


On line 15 we instantiate our :doc:`MPI Executor<../executor/mpi_executor>` class instance,
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
        "in": ["sim_id"],  # Generator input
        "out": [("x", float, (1,))],  # Name, type and size of data from gen_f
        "user": {
            "lb": np.array([1000]),  # User parameters for the gen_f
            "ub": np.array([3000]),
            "gen_batch_size": 8,
        },
    }

Our generation function will generate random numbers of particles, between
the ``"lb"`` and ``"ub"`` bounds, for our simulation function to evaluate via our
registered application.

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

 1. Override the MPIExecutor's detected MPI runner with ``'openmpi'``.
 2. Set the libEnsemble logger to print DEBUG messages.
 3. Save the History array and ``persis_info`` to files once libEnsemble completes.

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

        #!/usr/bin/env python
        import os
        import numpy as np
        from tutorial_forces import run_forces  # Sim func from current dir

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

        # Launch libEnsemble
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs
        )

        if is_manager:
            save_libE_output(H, persis_info, __file__, nworkers)


Simulation Function
-------------------

Our simulation function is where we'll use libEnsemble's executor to configure and submit
our application for execution. Once we've detected it has finished we'll send any results or
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

        # Submit our forces app for execution. Block until the task starts.
        task = exctr.submit(app_name="forces", app_args=args, wait_on_start=True)

        # Block until the task finishes
        task.wait(timeout=60)

Notably, we retrieve the generated number of particles and construct
an argument string for our launched application. We've retrieved our
previously-instantiated Executor instance from the class definition,
where it was automatically stored as an attribute.

When an application instance is submitted by one of libEnsemble's executors,
a :ref:`Task<task_tag>` object is returned that correlates with the launched app.
This object is roughly equivalent to a Python future, and can be polled, killed,
and evaluated in a variety of helpful ways. For now, we're satisfied with waiting
for the task to complete.

Since ``task.wait()`` blocks until the task completes, we can assume that afterward, any
results are now available to parse. Our application produces a ``forces[particles].stat``
file during runtime that contains either energy computations for every time-step
or a "kill" message if particles were lost. This last message indicates a failed
simulation.

To complete our simulation function, parse the last energy value from the output file into
a local output :ref:`History array<datastruct-history-array>`, and if successful,
set the simulation function's exit status :ref:`calc_status<datastruct-calc-status>`
to ``WORKER_DONE``. Otherwise, send back ``NAN`` and a ``TASK_FAILED`` status:

.. code-block:: python
    :linenos:

        # Stat file to check for bad runs
        statfile = "forces{}.stat".format(particles)

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
with launching applications, since behind the scenes libEnsemble evaluates default
MPI runners and available resources, and divides them among workers accordingly.

This completes our calling script and simulation function. Run libEnsemble with:

.. code-block:: bash

    $ python my_calling_script.py --comms local --nworkers [nworkers]

This may take up to a minute to complete. Output files, including ``forces.stat``
and files containing ``stdout`` and ``stderr`` content for each task should
appear in the current working directory. Overall workflow information
should appear in ``libE_stats.txt`` and ``ensemble.log`` as usual.

For example, my ``libE_stats.txt`` resembled::

  Worker     1: Gen no     1: gen Time: 0.001 Start: ... End: ... Status: Not set
  Worker     1: sim_id     0: sim Time: 3.201 Start: ... End: ... Status: Completed
  Worker     2: sim_id     1: sim Time: 3.208 Start: ... End: ... Status: Task Failed
  Worker     1: sim_id     2: sim Time: 0.228 Start: ... End: ... Status: Completed
  Worker     2: sim_id     3: sim Time: 0.236 Start: ... End: ... Status: Task Failed
  Worker     1: sim_id     4: sim Time: 0.229 Start: ... End: ... Status: Task Failed
  Worker     2: sim_id     5: sim Time: 0.233 Start: ... End: ... Status: Task Failed
  Worker     1: sim_id     6: sim Time: 0.227 Start: ... End: ... Status: Completed
  Worker     2: sim_id     7: sim Time: 0.228 Start: ... End: ... Status: Task Failed

Where ``status`` is set based on the simulation function's returned ``calc_status``.

My ``ensemble.log`` (on a ten-core laptop) resembled::

  [0]  ... libensemble.libE (INFO): Logger initializing: [workerID] precedes each line. [0] = Manager
  [0]  ... libensemble.libE (INFO): libE version v0.8.0+dev
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

Advanced Exercises
^^^^^^^^^^^^^^^^^^

TODO

.. _here: https://raw.githubusercontent.com/Libensemble/libensemble/master/libensemble/tests/scaling_tests/forces/forces.c
.. _examples/tutorials/forces_with_executor: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/forces_with_executor
