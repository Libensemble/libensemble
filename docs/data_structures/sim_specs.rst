.. _datastruct-sim-specs:

Simulation Specs
================

Used to specify the simulation function, its inputs and outputs, and user data.

.. tab-set::

    .. tab-item:: Standardized (gest-api)

        .. code-block:: python
          :linenos:

          from libensemble import SimSpecs
          from gest_api.vocs import VOCS
          from my_package import my_sim_callable

          vocs = VOCS(
              variables={"x": [-3.0, 3.0]},
              objectives={"y": "MINIMIZE"},
          )

          sim_specs = SimSpecs(
              simulator=my_sim_callable,
              vocs=vocs,
          )
          ...

    .. tab-item:: Classic (sim_f)

        .. code-block:: python
          :linenos:

          from libensemble import SimSpecs
          from simulator import sim_find_sine

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
