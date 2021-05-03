==================================
Executor with Electrostatic Forces
==================================

This tutorial highlights libEnsemble's capability to execute
and monitor external scripts or user applications within simulation or generator
functions using the :doc:`executor<../executor/overview>`. In this tutorial,
our calling script registers an external C executable that simulates
electrostatic forces between a collection of particles. The ``sim_f``
routine then launches and polls this executable.

It is possible to use ``subprocess`` calls from Python to issue
commands such as ``jsrun`` or ``aprun`` to run applications. Unfortunately,
hard-coding such commands within user scripts isn't portable.
Furthermore, many systems like Argonne's :doc:`Theta<../platforms/theta>` do not
allow libEnsemble to submit additional tasks from the compute nodes. On these
systems a proxy launch mechanism (such as Balsam) is required.
libEnsemble's executor was developed to directly address such issues.

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
    :emphasize-lines: 22

    #!/usr/bin/env python
    import os
    import numpy as np
    from forces_simf import run_forces  # Sim func from current dir

    from libensemble.libE import libE
    from libensemble.gen_funcs.sampling import uniform_random_sample
    from libensemble.tools import parse_args, add_unique_random_streams
    from libensemble.executors.mpi_executor import MPIExecutor

    nworkers, is_manager, libE_specs, _ = parse_args()  # Convenience function

    # Create executor and register sim to it
    exctr = MPIExecutor()  # Use auto_resources=False to oversubscribe

    # Create empty simulation input directory
    if not os.path.isdir('./sim'):
        os.mkdir('./sim')

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), 'forces.x')
    exctr.register_calc(full_path=sim_app, calc_type='sim')

On line 4 we import our not-yet-written ``sim_f``. We also import necessary
libEnsemble components and some :doc:`convenience functions<../utilities>`.
For example, our script can use the number of workers (``nworkers``), a boolean
determining if the process is the manager process (``is_manager``), and a default
:ref:`libE_specs<datastruct-libe-specs>` with a call to the ``parse_args()``
convenience function.

Next we define our executor class instance. This instance can be customized
with many of the settings defined :doc:`here<../executor/mpi_executor>`.
We'll register our simulation with the executor and use the same
instance within our ``sim_f``.

libEnsemble can perform and write every simulation (within the ensemble) in a
separate directory for organization and potential I/O benefits. In this example,
libEnsemble copies a source directory and its contents to create these simulation
directories. For our purposes, an empty directory ``./sim`` is sufficient.

Next define the :ref:`sim_specs<datastruct-sim-specs>` and
:ref:`gen_specs<datastruct-gen-specs>` data structures:

.. code-block:: python
    :linenos:

    # State the sim_f, its arguments, output, and parameters (and their sizes)
    sim_specs = {'sim_f': run_forces,         # sim_f, imported above
                 'in': ['x'],                 # Name of input for sim_f
                 'out': [('energy', float)],  # Name, type of output from sim_f
                 'user': {'simdir_basename': 'forces',  # User parameters for sim_f
                          'cores': 2,
                          'sim_particles': 1e3,
                          'sim_timesteps': 5,
                          'sim_kill_minutes': 10.0,
                          'particle_variance': 0.2,
                          'kill_rate': 0.5}
                 }

    # State the gen_f, its arguments, output, and necessary parameters.
    gen_specs = {'gen_f': uniform_random_sample,  # Generator function
                 'in': ['sim_id'],                # Generator input
                 'out': [('x', float, (1,))],     # Name, type and size of data from gen_f
                 'user': {'lb': np.array([0]),            # User parameters for gen_f
                          'ub': np.array([32767]),
                          'gen_batch_size': 1000,
                          'batch_mode': True,
                          'num_active_gens': 1,
                          }
                 }

These dictionaries configure our generation function ``gen_f`` and our simulation
function ``sim_f``, respectively, as the ``uniform_random_sample`` and
``run_forces`` functions. Our ``gen_f`` will generate random seeds when
initializing each ``sim_f`` call.

After some additions to ``libE_specs`` and defining our ``exit_criteria`` and
``persis_info``, our script calls the main
:doc:`libE<../libe_module>` routine:

 .. code-block:: python
    :linenos:

    libE_specs['save_every_k_gens'] = 1000  # Save every K steps
    libE_specs['sim_input_dir'] = './sim'   # Sim dir to be copied for each worker

    exit_criteria = {'sim_max': 8}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info=persis_info, libE_specs=libE_specs)

Simulation Function
-------------------

Our ``sim_f`` is where we'll use libEnsemble's executor to configure and submit
for execution our compiled simulation code. We will poll this task's state while
it runs, and once we've detected it has finished we will send any results or
exit statuses back to the manager.

Create another Python file named ``forces_simf.py`` containing:

.. code-block:: python
    :linenos:

    import os
    import time
    import numpy as np

    from libensemble.executors.executor import Executor
    from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

    MAX_SEED = 32767

    def perturb(particles, seed, max_fraction):
        """Modify particle count"""
        seed_fraction = seed/MAX_SEED
        max_delta = particles * max_fraction
        delta = seed_fraction * max_delta
        delta = delta - max_delta/2  # translate so -/+
        new_particles = particles + delta
        return int(new_particles)

    def read_last_line(filepath):
        """Read last line of statfile"""
        try:
            with open(filepath, 'rb') as fh:
                line = fh.readlines()[-1].decode().rstrip()
        except Exception:
            line = ""  # In case file is empty or not yet created
        return line

We use libEnsemble's message number tags to communicate the worker's status to
the manager. For testing purposes, the ``perturb()`` function randomizes the
resources used for each calculation. The second function parses
forces values and statuses in the ``.stat`` file produced by our compiled code.
Now we can write the actual ``sim_f``. We'll first write the function definition,
extract our parameters from ``sim_specs``, define a random seed, and use
``perturb()`` to randomize our particle counts.

.. code-block:: python
    :linenos:

    def run_forces(H, persis_info, sim_specs, libE_info):
        calc_status = 0

        x = H['x']
        sim_particles = sim_specs['user']['sim_particles']
        sim_timesteps = sim_specs['user']['sim_timesteps']
        time_limit = sim_specs['user']['sim_kill_minutes'] * 60.0

        cores = sim_specs['user'].get('cores', None)
        kill_rate = sim_specs['user'].get('kill_rate', 0)
        particle_variance = sim_specs['user'].get('particle_variance', 0)

        seed = int(np.rint(x[0][0]))

        # To give a random variance of work-load
        sim_particles = perturb(sim_particles, seed, particle_variance)

Next we will instantiate our executor and submit our registered application for
execution.

.. code-block:: python
    :linenos:
    :emphasize-lines: 2,9,10,12,13

        # Use pre-defined executor object
        exctr = Executor.executor

        # Arguments for our registered simulation
        args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed) + ' ' + str(kill_rate)

        # Submit our simulation for execution.
        if cores:
            task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args,
                                stdout='out.txt', stderr='err.txt', wait_on_run=True)
        else:
            task = exctr.submit(calc_type='sim', app_args=args, stdout='out.txt',
                                stderr='err.txt', wait_on_run=True)

In each executor ``submit()`` routine, we define the type of calculation being
performed, optionally the number of processors to run the task on, additional
arguments for the simulation code, and files for ``stdout`` and ``stderr``
output. The ``wait_on_run`` argument pauses ``sim_f`` execution until the task
is confirmed to be running. See the :doc:`docs<../executor/mpi_executor>`
for more information about these and other options.

The rest of our ``sim_f`` polls the :ref:`task<task_tag>`'s
dynamically updated attributes for its status, determines if a successful
run occurred after the task completes, then formats and returns the output data
to the manager.

We can poll the task and kill it in certain circumstances:

.. code-block:: python
    :linenos:
    :emphasize-lines: 7,10-12,15

        # Stat file to check for bad runs
        statfile = 'forces.stat'
        filepath = os.path.join(task.workdir, statfile)
        line = None

        poll_interval = 1
        while not task.finished :
            line = read_last_line(filepath)  # Parse some output from the task
            if line == "kill":
                task.kill()
            elif task.runtime > time_limit:
                task.kill()
            else:
                time.sleep(poll_interval)
                task.poll()                   # updates the task's attributes

Once our task finishes, adjust ``calc_status`` (our "exit code") and report to the
user based on the task's final state:

.. code-block:: python
    :linenos:
    :emphasize-lines: 1-3,7,8,10,11,14

        if task.finished:
            if task.state == 'FINISHED':
                print("Task {} completed".format(task.name))
                calc_status = WORKER_DONE
                if read_last_line(filepath) == "kill":
                    print("Warning: Task complete but marked bad (kill flag in forces.stat)")
            elif task.state == 'FAILED':
                print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
                calc_status = TASK_FAILED
            elif task.state == 'USER_KILLED':
                print("Warning: Task {} has been killed".format(task.name))
                calc_status = WORKER_KILL
            else:
                print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))

Load output data from our task and return to the libEnsemble manager:

.. code-block:: python
    :linenos:

        time.sleep(0.2) # Small buffer to guarantee data has been written
        try:
            data = np.loadtxt(filepath)
            final_energy = data[-1]
        except Exception:
            final_energy = np.nan

        outspecs = sim_specs['out']
        output = np.zeros(1, dtype=outspecs)
        output['energy'][0] = final_energy

        return output, persis_info, calc_status

This completes our ``sim_f`` and calling script. Run libEnsemble with:

.. code-block:: bash

    $ python my_calling_script.py --comms local --nworkers 4

This may take about a minute to complete. Output should appear in a new
directory ``./ensemble``, with sub-directories labeled by ``sim_id`` and worker.

The following optional lines parse and display some output:

.. code-block:: python
    :linenos:

    import os

    for dir in os.listdir('./ensemble'):
        with open(os.path.join('./ensemble', dir, 'out.txt')) as f:
            out = f.readlines()
        print(dir + ':')
        for line in out:
            print(line)
        print('-'*60)

Executor Variants
-----------------

libEnsemble features two variants of its executor that perform identical
functions, but are designed for running on different systems. For most uses,
the MPI variant will be satisfactory. However, some systems, such as ALCF's Theta
do not support MPI launches from compute nodes. On these systems libEnsemble is
run either on launch nodes or uses a proxy launch mechanism to submit
tasks from compute nodes. One such mechanism is a scheduling utility called
Balsam_ which runs on a separate node. The Balsam Executor variant interacts
with Balsam for this purpose. The only user-facing difference between the two is
which executor is imported and called within a calling script.

.. _here: https://raw.githubusercontent.com/Libensemble/libensemble/master/libensemble/tests/scaling_tests/forces/forces.c
.. _Balsam: https://balsam.readthedocs.io/en/latest/
