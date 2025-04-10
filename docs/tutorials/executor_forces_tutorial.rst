================================
Ensemble with an MPI Application
================================

This tutorial highlights libEnsemble's capability to portably execute
and monitor external scripts or user applications within simulation or generator
functions using the :doc:`executor<../executor/overview>`.

|Open in Colab|

The calling script registers a compiled executable that simulates
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

    mpicc -O3 -o forces.x forces.c -lm

Alternative build lines for different platforms can be found in the build_forces.sh_
file in the same directory.

Calling Script
--------------

Complete scripts for this example can be found in the forces_simple_ directory.

Let's begin by writing our calling script to specify our simulation and
generation functions and call libEnsemble. Create a Python file called
`run_libe_forces.py` containing:

.. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial.py
    :language: python
    :linenos:
    :end-at: ensemble = Ensemble

We first instantiate our :doc:`MPI Executor<../executor/mpi_executor>`.
Registering an application is as easy as providing the full file-path and giving
it a memorable name. This Executor will later be used within our simulation
function to launch the registered app.

The last line in the above codeblock initializes the ensemble. The :meth:`parse_args<tools.parse_args>`
parameter is used to read `comms` and `nworkers` from the command line. This sets
the respective `libE_specs` options.

Next, we will add basic configuration for the ensemble. As one worker will run a persistent
generator, we calculate the number of workers that need resources to run simulations.
We also set `sim_dirs_make` so that a directory is created for each simulation. This
helps organize output and also helps prevent workers from overwriting previous results.

.. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial.py
    :language: python
    :linenos:
    :start-at: nsim_workers = ensemble.nworkers
    :end-at: )
    :lineno-start: 28

Next we define the :ref:`sim_specs<datastruct-sim-specs>` and
:ref:`gen_specs<datastruct-gen-specs>`. Recall that these are used to specify
to libEnsemble what user functions and input/output fields to
expect, and also to parameterize user functions:

.. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial.py
    :language: python
    :linenos:
    :start-at: ensemble.sim_specs = SimSpecs(
    :end-at: gen_specs_end_tag
    :lineno-start: 37

Next, configure an allocation function, which starts the one persistent
generator and farms out the simulations. We also tell it to wait for all
simulations to return their results, before generating more parameters.

.. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial.py
    :language: python
    :linenos:
    :start-at: ensemble.alloc_specs = AllocSpecs
    :end-at: )
    :lineno-start: 55

Now we set :ref:`exit_criteria<datastruct-exit-criteria>` to
exit after running eight simulations.

We also give each worker a seeded random stream, via the
:ref:`persis_info<datastruct-persis-info>`  option.
These can be used for random number generation if required.

Finally we :doc:`run<../libe_module>` the ensemble.

.. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial.py
    :language: python
    :linenos:
    :start-at: Instruct libEnsemble
    :end-at: ensemble.run()
    :lineno-start: 62

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

      .. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial_2.py
          :language: python
          :start-at: from libensemble import Ensemble, logger
          :end-at: logger.set_level("DEBUG")

   **Soln 2.** This can also be specified via :ref:`platform_specs<datastruct-platform-specs>` option.

      .. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial_2.py
          :language: python
          :start-at: Initialize MPI Executor
          :end-at: exctr = MPIExecutor

   **Soln 3.** Set ``async_return`` to *True* in the allocation .

      .. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial_2.py
          :language: python
          :start-at: # Starts one persistent generator
          :end-at: )

   **Soln 4.** End your script in the following manner to save the output based
   on the name of the calling script. You can give any string in place of ``__file__``.

      .. literalinclude:: ../../libensemble/tests/functionality_tests/test_executor_forces_tutorial_2.py
          :language: python
          :start-at: Run ensemble
          :end-at: save_output

Simulation Function
-------------------

Our simulation function is where we'll use libEnsemble's executor to configure and submit
our application for execution. We'll poll this task's state while
it runs, and once we've detected it has finished we'll send any results or
exit statuses back to the manager.

Create another Python file named ``forces_simf.py`` containing the following
for starters:

.. literalinclude:: ../../libensemble/tests/functionality_tests/forces_simf.py
    :language: python
    :linenos:
    :end-at: task.wait()

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

.. literalinclude:: ../../libensemble/tests/functionality_tests/forces_simf.py
    :language: python
    :linenos:
    :lineno-start: 27
    :start-at: Try loading final

``calc_status`` will be displayed in the ``libE_stats.txt`` log file.

That's it! As can be seen, with libEnsemble, it's relatively easy to get started
with launching applications.

Running the example
-------------------

This completes our calling script and simulation function. Run libEnsemble with:

.. code-block:: bash

    python run_libe_forces.py --nworkers [nworkers]

where ``nworkers`` is one more than the number of concurrent simulations.

Output files (including ``forces.stat`` and files containing ``stdout`` and
``stderr`` content for each task) should appear in the current working
directory. Overall workflow information should appear in ``libE_stats.txt``
and ``ensemble.log`` as usual.

.. dropdown:: **Example run / output**

   For example, after running:

   .. code-block:: bash

       python run_libe_forces.py --nworkers 3

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

Running the generator on the manager
------------------------------------

As of version 1.3.0, the generator can be run on a thread on the manager,
using the :ref:`libE_specs<datastruct-libe-specs>` option **gen_on_manager**.

Change the libE_specs as follows.

   .. code-block:: python
    :linenos:
    :lineno-start: 28

    nsim_workers = ensemble.nworkers

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
        ensemble_dir_path="./test_executor_forces_tutorial",
    )

When running set ``nworkers`` to the number of workers desired for running simulations.
E.g., Instead of:

.. code-block:: bash

    python run_libe_forces.py --nworkers 5

use:

.. code-block:: bash

    python run_libe_forces.py --nworkers 4

Note that as the generator random number seed will be zero instead of one, the checksum will change.

For more information see :ref:`Running generator on the manager<gen-on-manager>`.

Running forces application with input file
------------------------------------------

Many applications read an input file instead of being given parameters directly on the run line.

forces_simple_with_input_file_ directory contains a variant of this example, where a templated
input file is parameterized for each evaluation.

This requires **jinja2** to be installed::

    pip install jinja2

The file ``forces_input`` contains the following (remember we are using particles
as seed also for simplicity)::

    num_particles = {{particles}}
    num_steps = 10
    rand_seed = {{particles}}

libEnsemble will copy this input file to each simulation directory.

The ``sim_f`` uses the following function to customize the input file with the parameters
for the current simulation.

.. code-block:: python

    def set_input_file_params(H, sim_specs, ints=False):
        """
        This is a general function to parameterize the input file with any inputs
        from sim_specs["in"]

        Often sim_specs_in["x"] may be multi-dimensional, where each dimension
        corresponds to a different input name in sim_specs["user"]["input_names"]).
        Effectively an unpacking of "x"
        """
        input_file = sim_specs["user"]["input_filename"]
        input_values = {}
        for i, name in enumerate(sim_specs["user"]["input_names"]):
            value = int(H["x"][0][i]) if ints else H["x"][0][i]
            input_values[name] = value
        with open(input_file, "r") as f:
            template = jinja2.Template(f.read())
        with open(input_file, "w") as f:
            f.write(template.render(input_values))

This is called in the simulation function as follows.

.. code-block:: python

    def run_forces(H, persis_info, sim_specs, libE_info):
        """Runs the forces MPI application reading input from file"""

        calc_status = 0

        input_file = sim_specs["user"]["input_filename"]
        set_input_file_params(H, sim_specs, ints=True)

        # Retrieve our MPI Executor
        exctr = libE_info["executor"]

        # Submit our forces app for execution.
        task = exctr.submit(app_name="forces")  # app_args removed

        # Block until the task finishes
        task.wait(timeout=60)

Notice that we convert the parameters to integers in this example.

The calling script then specifies the templated input file as follows.

.. code-block:: python
    :linenos:
    :lineno-start: 30
    :emphasize-lines: 1,7,14

    input_file = "forces_input"

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        sim_dirs_make=True,
        sim_dir_copy_files=[input_file],
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        inputs=["x"],
        outputs=[("energy", float)],
        user={"input_filename": input_file, "input_names": ["particles"]},
    )

.. Note sphinx does not adjust for lineno-start

Line 36 tells the templated input file to be copied to each simulation directory.

An alternative is to use ``sim_input_dir``, which gives the name of a directory
that may contain multiple files and will be used as the base of each simulation
directory.

Line 43 gives the input file name and the name of each parameter to the simulation
function.

Multiple parameters
^^^^^^^^^^^^^^^^^^^

In our case, the only parameter name is ``x``. However, in some cases, ``x``
(as defined by ``sim_specs["in"]``) may be multi-dimensional, where each
component has a different parameter name (e.g., "x", "y"). For example, if the
input file were::

    num_particles = {{particles}}
    num_steps = {{nsteps}}
    rand_seed = {{seed}}

then line 43 would be:

.. code-block:: python

    user = {"input_filename": input_file, "input_names": ["particles", "nsteps", "seed"]}

and ``gen_specs`` would contain something similar to:

.. code-block:: python
    :linenos:
    :lineno-start: 46
    :emphasize-lines: 5

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],
        persis_in=["sim_id"],
        outputs=[("x", float, 3)],
        ...,
    )

libEnsemble uses a convention of a multi-dimensional ``x`` in generator functions. However,
these parameters can also be specified as different variables with corresponding modification
to generator and simulator functions.

.. _examples/tutorials/forces_with_executor: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/forces_with_executor
.. _build_forces.sh: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/scaling_tests/forces/forces_app/build_forces.sh
.. _forces_app: https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/forces/forces_app
.. _forces_simple: https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/forces/forces_simple
.. _forces_simple_with_input_file: https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/forces/forces_simple_with_input_file
.. _GitHub: https://github.com/Libensemble/libensemble/issues
.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target:  http://colab.research.google.com/github/Libensemble/libensemble/blob/develop/examples/tutorials/forces_with_executor/forces_tutorial_notebook.ipynb
