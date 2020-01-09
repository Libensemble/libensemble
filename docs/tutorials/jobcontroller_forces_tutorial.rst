============================================================
Using the Job Controller to launch Electrostatic Simulations
============================================================

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

This simulation source ``forces.c`` can be obtained directly from the libEnsemble
 repository here_.

Assuming MPI and its C compiler ``mpicc`` are installed on your system, compile
``forces.c`` into an executable (``forces.x``) with:

.. code-block:: bash

    $ mpicc -O3 -o forces.x forces.c -lm



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
