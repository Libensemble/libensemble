Mock simulation evaluations (re-run using history file)
-------------------------------------------------------

.. role:: underline
   :class: underline

.. automodule:: mock_sim
   :members:
   :undoc-members:

.. dropdown:: :underline:`mock_sim.py`

   .. literalinclude:: ../../../libensemble/sim_funcs/mock_sim.py
      :language: python
      :linenos:

.. dropdown:: :underline:`Example usage`

   This test runs two repetitions. The first ensemble dumps a history file, the second replays
   the first run using the mock sim with the history file.

   .. literalinclude:: ../../../libensemble/tests/functionality_tests/test_uniform_sampling.py
      :language: python
      :linenos:
