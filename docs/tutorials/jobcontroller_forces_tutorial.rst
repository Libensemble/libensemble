=====================================
Job Controller & Electrostatic Forces
=====================================

This tutorial describes and teaches libEnsemble's additional capability to launch and
monitor external scripts or user applications within simulation or generator
functions using the :ref:`Job Controller<jobcontroller_index>`. In this tutorial,
we register an external C simulation for particle electrostatic forces in
our calling script then launch and poll it within our ``sim_f``. This allows us
to scale our C simulation using libEnsemble without rewriting it as a Python
``sim_f``.

While traditional Python ``subprocess`` calls or high-performance
mechanisms like ``jsrun`` or ``aprun`` can successfully submit applications for
processing, hardcoding these routines as-is into a ``sim_f`` isn't portable.
Furthermore, many systems like Argonne's :doc:`Theta<../platforms/theta>` do not
support submitting additional jobs from compute nodes. libEnsemble's job
controller was developed to directly address these issues.

Getting Started
---------------

The simulation source code ``forces.c`` can be obtained directly from the
libEnsemble repository here_.

Assuming MPI and its C compiler ``mpicc`` are installed on your system, compile
``forces.c`` into an executable (``forces.x``) with:

.. code-block:: bash

    $ mpicc -O3 -o forces.x forces.c -lm

Calling Script
--------------

Lets begin by writing our calling script to parameterize our simulation and generation
functions and call libEnsemble. Create an empty Python file and type (or copy and
paste...) the following:

.. code-block:: python
    :linenos:
    :emphasize-lines: 22

    #!/usr/bin/env python
    import os
    import numpy as np
    from forces_simf import run_forces  # Sim func from current dir

    from libensemble.libE import libE
    from libensemble.gen_funcs.sampling import uniform_random_sample
    from libensemble.utils import parse_args, add_unique_random_streams
    from libensemble.mpi_controller import MPIJobController

    nworkers, is_master, libE_specs, _ = parse_args()  # Convenience function

    # Create job_controller and register sim to it
    jobctrl = MPIJobController()  # Use auto_resources=False to oversubscribe

    # Create empty simulation input directory
    if not os.path.isdir('./sim'):
        os.mkdir('./sim')

    # Register simulation executable with job controller
    sim_app = os.path.join(os.getcwd(), 'forces.x')
    jobctrl.register_calc(full_path=sim_app, calc_type='sim')

On line 4 we import our not-yet-written ``sim_f``. We also import necessary
libEnsemble components and some :doc:`convenience functions<../utilities>`.

We can quickly define the number of workers, if the current process is the master
process and :ref:`libE_specs<datastruct-libe-specs>` with a call to the
``parse_args()`` convenience function.

libEnsemble can perform and write every simulation "step" in a separate directory
for organization and potential I/O speed benefits. libEnsemble copies a source
directory and its contents to create these simulation directories.
For our purposes, an empty directory ``./sim`` is sufficient.

Next we have our :ref:`sim_specs<datastruct-sim-specs>` and :ref:`gen_specs<datastruct-gen-specs>`:

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
function ``sim_f``, referred to as ``uniform_random_sample`` and ``run_forces``,
respectively. Our ``gen_f`` will be used primarily to generate random seeds for
initializing the simulation within our ``sim_f``.

After some additions to ``libE_specs`` and defining our ``exit_criteria`` and
``persis_info``, we conclude our calling script with a call to the main
:doc:`libE<../libe_module>` routine:

 .. code-block:: python
    :linenos:

    libE_specs['save_every_k_gens'] = 1000  # Save every K steps
    libE_specs['sim_dir'] = './sim'         # Sim dir to be copied for each worker

    exit_criteria = {'sim_max': 8}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info=persis_info, libE_specs=libE_specs)

Simulation Function:
--------------------

Our simulation function is where we'll configure and launch our compiled simulation
code as a job using libEnsemble's Job Controller. We will poll this job's state,
and send any results or exit statuses back to the manager.

Create another empty Python file and start by writing or pasting the following:

.. code-block:: python
    :linenos:

    import os
    import time
    import numpy as np

    from libensemble.controller import JobController
    from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, JOB_FAILED

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

We use libEnsemble's built-in message number tags in place of indescriptive integers.
``perturb()`` is used to randomize the work-load of particle calculations for testing
purposes. Our compiled code outputs forces values and statuses with a ``.stat`` file,
the second function parses that file.


Job Controller Variants
-----------------------

libEnsemble features two variants of its Job Controller that perform identical
functions, but are meant to run on different system architectures. For most uses,
the MPI variant will be satisfactory. Some systems like Theta, mentioned previously,
require an additional scheduling utility called Balsam_ running on a separate node
for job submission to function properly. The Balsam Job Controller variant interacts
with Balsam for this purpose. The only user-facing difference between the two is
which controller is imported and called within a calling script.


.. _here: https://raw.githubusercontent.com/Libensemble/libensemble/master/libensemble/tests/scaling_tests/forces/forces.c
.. _Balsam: https://balsam.readthedocs.io/en/latest/
