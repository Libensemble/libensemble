=================================================
Electrostatic Simulations Job Controller Tutorial
=================================================

This tutorial demonstrates libEnsemble's additional capability to launch and
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

The simulation source code ``forces.c`` can be obtained directly from the libEnsemble
 repository here_.

Assuming MPI and its C compiler ``mpicc`` are installed on your system, compile
``forces.c`` into an executable (``forces.x``) with:

.. code-block:: bash

    $ mpicc -O3 -o forces.x forces.c -lm

Calling script
--------------

Lets begin by writing our calling script to parameterize our simulation and generation
functions and call libEnsemble. Create an empty Python file and type (or copy and
paste...) the following:

.. code-block:: python
    :linenos:
    :emphasize-lines: 23

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

On line 4 we import our not-yet-written ``sim_f``. The next four libEnsemble
statements import the primary :doc:`libE<../libe_module>` function, our ``gen_f``,
two convenience functions, and the job controller.

To quickly define and populate the number of workers, if the current process is
the master process, and ``libE_specs``, we include a call to ``parse_args()``.
We next create a job controller object instance.

libEnsemble has the capability to perform and write every simulation "step" in
a separate directory for organizational and potential I/O speed benefits. libEnsemble
copies a source directory and its contents to create these simulation directories.
For our purposes, an empty directory ``./sim`` is sufficient.

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
