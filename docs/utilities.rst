Convenience Tools and Functions
===============================

.. tab-set::

   .. tab-item:: Setup Helpers

      .. automodule:: tools
         :members:
         :no-undoc-members:

   .. tab-item:: Persistent Helpers

      .. _p_gen_routines:

      These routines are commonly used within persistent generator functions
      such as ``persistent_aposmm`` in ``libensemble/gen_funcs/`` for intermediate
      communication with the manager. Persistent simulator functions are also supported.

      .. automodule:: persistent_support
         :members:
         :no-undoc-members:

   .. tab-item:: Allocation Helpers

      These routines are used within custom allocation functions to help prepare ``Work``
      structures for workers. See the routines within ``libensemble/alloc_funcs/`` for
      examples.

      .. automodule:: alloc_support
         :members:
         :no-undoc-members:
