======================
Writing User Functions
======================

User functions typically require only some familiarity with NumPy_, but if they conform to
the :ref:`user function APIs<user_api>`, they can incorporate methods from machine-learning,
mathematics, resource management, or other libraries/applications.

These guides describe common development patterns and optional components:

.. toctree::
   :maxdepth: 2
   :caption: Writing User Functions

   generator
   simulator
   allocator
   sim_gen_alloc_api

.. toctree::
   :maxdepth: 2
   :caption: Useful Data Structures

   calc_status
   work_dict
   worker_array

.. _NumPy: http://www.numpy.org
