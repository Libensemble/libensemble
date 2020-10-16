======================
Writing User Functions
======================

libEnsemble coordinates ensembles of calculations performed by three main
functions: a :ref:`Generator Function<api_gen_f>`, a :ref:`Simulator Function<api_sim_f>`,
and an :ref:`Allocation Function<api_alloc_f>`, or ``gen_f``, ``sim_f``, and
``alloc_f`` respectively. These are all referred to as User Functions. Although
libEnsemble includes several ready-to-use User Functions like
:doc:`APOSMM<../examples/aposmm>`, it's expected many users will write their own or
adjust included functions for their own use-cases.
These guides describe common development patterns and optional components for
each kind of User Function.

.. toctree::
   :maxdepth: 2

   generator
   simulator
   allocator
