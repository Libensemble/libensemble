.. _datastruct-sim-specs:

Simulation Specs
================

Used to specify the simulation function, its inputs and outputs, and user data.

Can be constructed and passed to libEnsemble as a Python class or a dictionary. When provided as a Python class,
all data is validated immediately on instantiation.

.. tab-set::

  .. tab-item:: class

    .. code-block:: python
        :linenos:

        ...
        import numpy as np
        from libensemble import SimSpecs
        from simulator import sim_find_sine

        ...

        sim_specs = SimSpecs(
            sim_f=sim_find_sine,
            inputs=["x"],
            out=[("y", float)],
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

  .. tab-item:: dict

    .. code-block:: python
        :linenos:

        ...
        import numpy as np
        from simulator import six_hump_camel

        ...

        sim_specs = {
            "sim_f": six_hump_camel,
            "in": ["x"],
            "out": [("y", float)],
            "user": {"batch": 1234},
        }
        ...

    - test_uniform_sampling.py_ has a :class:`sim_specs<libensemble.specs.SimSpecs>`  that declares
      the name of the ``"in"`` field variable, ``"x"`` (as specified by the
      corresponding generator ``"out"`` field ``"x"`` from the :ref:`gen_specs
      example<gen-specs-example1>`).  Only the field name is required in
      ``sim_specs["in"]``.

    ..  literalinclude:: ../../libensemble/tests/functionality_tests/test_uniform_sampling.py
        :start-at: sim_specs
        :end-before: end_sim_specs_rst_tag

    - run_libe_forces.py_ has a longer :class:`sim_specs<libensemble.specs.SimSpecs>` declaration with a number of
      user-specific fields. These are given to the corresponding sim_f, which
      can be found at forces_simf.py_.

    ..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_adv/run_libe_forces.py
        :start-at: sim_f
        :end-before: end_sim_specs_rst_tag

.. _forces_simf.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_simf.py
.. _run_libe_forces.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py
