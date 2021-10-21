Convenience Tools and Functions
===============================

Calling Script Function Support
-------------------------------

.. automodule:: tools
   :members:
   :no-undoc-members:

Persistent Function Support
---------------------------
.. _p_gen_routines:

These routines are commonly used within persistent generator functions
like ``persistent_aposmm`` in ``libensemble/gen_funcs/`` for intermediate
communication with the manager. Persistent simulator functions are also supported.

.. automodule:: persistent_support
   :members:
   :no-undoc-members:

Allocation Function Support
---------------------------

These routines are used within custom allocation functions to help prepare Work
structures for workers. See the routines within ``libensemble/alloc_funcs/`` for
examples.

.. automodule:: alloc_support
   :members:
   :no-undoc-members:

Consensus Subroutines
---------------------

.. automodule:: consensus_subroutines
   :members:
   :no-undoc-members:
