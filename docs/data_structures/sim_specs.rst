.. _datastruct-sim-specs:

Simulation Specs
================

Used to specify the simulation, its inputs and outputs, and user data.

.. code-block:: python
  :linenos:

  ...
  from libensemble import SimSpecs
  from simulator import sim_find_sine

  ...

  sim_specs = SimSpecs(
      sim_f=sim_find_sine,
      inputs=["x"],
      outputs=[("y", float)],
      user={"batch": 1234},
  )
  ...

.. autopydantic_model:: libensemble.specs.SimSpecs
  :model-show-json: False
  :model-show-config-member: False
  :model-show-config-summary: False
  :model-show-validator-members: False
  :model-show-validator-summary: False
  :field-list-validators: False


.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py
