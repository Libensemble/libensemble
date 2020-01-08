=============================
Electrostatic Forces Tutorial
=============================

This tutorial demonstrates libEnsemble's additional capability to launch and
monitor external scripts or user applications within simulation or generator
functions using the :ref:`Job Controller<jobcontroller_index>`. In this tutorial,
we register an external C simulation for inter-particle electrostatic forces in
our calling script then launch and poll it within our ``sim_f``. This allows us
to scale our C simulation using libEnsemble without having to rewriting it as
a Python ``sim_f``.

Typically, while traditional Python ``subprocess`` calls or high-performance
mechanisms like ``jsrun`` or ``aprun`` can successfully submit applications for
processing, hardcoding these routines as-is into a ``sim_f`` isn't portable.
Furthermore, many systems like Argonne's :doc:`Theta<../platforms/theta>` do not
support submitting additional jobs from compute nodes. libEnsemble's job controller
was developed to directly address these issues.

This tutorial will introduce each component of the job controller in turn.
