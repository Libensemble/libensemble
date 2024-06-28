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

   .. tab-item:: Live Data

      These classes provide a means to capture and display data during a workflow run.
      Users may provide an initialized object via ``libE_specs["live_data"]``. For example::

        from libensemble.tools.live_data.plot2n import Plot2N
        libE_specs["live_data"] = Plot2N(plot_type='2d')

      .. automodule:: libensemble.tools.live_data.live_data
         :members:

      .. automodule:: plot2n
         :members: Plot2N
         :show-inheritance:
