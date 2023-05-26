.. _datastruct-gen-specs:

Generator Specs
===============

Used to specify the generator function, its inputs and outputs, and user data.

Can be constructed and passed to libEnsemble as a Python class or a dictionary. When provided as a Python class,
all data is validated immediately on instantiation.

.. tab-set::

  .. tab-item:: class

    .. code-block:: python
        :linenos:

        ...
        import numpy as np
        from libensemble import GenSpecs
        from generator import gen_random_sample

        ...

        gen_specs = GenSpecs(
            gen_f=gen_random_sample,
            out=[("x", float, (1,))],
            user={
                "lower": np.array([-3]),
                "upper": np.array([3]),
                "gen_batch_size": 5,
            },
        )
        ...

    .. autopydantic_model:: libensemble.specs.GenSpecs
      :model-show-json: False
      :model-show-config-member: False
      :model-show-config-summary: False
      :model-show-validator-members: False
      :model-show-validator-summary: False
      :field-list-validators: False

  .. tab-item:: dict

    .. code-block:: python
        :linenos:

        gen_specs = {
            "gen_f": gen_random_sample,
            "out": [("x", float, (1,))],
            "user": {
                "lower": np.array([-3]),
                "upper": np.array([3]),
                "gen_batch_size": 5,
            },
        }

    .. seealso::

      .. _gen-specs-example1:

      - test_uniform_sampling.py_:
        the generator function ``uniform_random_sample`` in sampling.py_ will generate 500 random
        points uniformly over the 2D domain defined by ``gen_specs["ub"]`` and
        ``gen_specs["lb"]``.

      ..  literalinclude:: ../../libensemble/tests/functionality_tests/test_uniform_sampling.py
          :start-at: gen_specs
          :end-before: end_gen_specs_rst_tag

    .. seealso::

        - test_persistent_aposmm_nlopt.py_ shows an example where ``gen_specs["in"]`` is empty, but
          ``gen_specs["persis_in"]`` specifies values to return to the persistent generator.

        - test_persistent_aposmm_with_grad.py_ shows a similar example where an ``H0`` is used to
          provide points from a previous run. In this case, ``gen_specs["in"]`` is populated to provide
          the generator with data for the initial points.

        - In some cases you might be able to give different (perhaps fewer) fields in ``"persis_in"``
          than ``"in"``; you may not need to give ``x`` for example, as the persistent generator
          already has ``x`` for those points. See `more example uses`_ of ``persis_in``.

.. note::

  * In all interfaces, custom fields should only be placed in ``"user"``
  * Generator ``"out"`` fields typically match Simulation ``"in"`` fields, and vice-versa.

.. _sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/gen_funcs/sampling.py
.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py
.. _test_persistent_aposmm_nlopt.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_nlopt.py
.. _test_persistent_aposmm_with_grad.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py
.. _more example uses: https://github.com/Libensemble/libensemble/wiki/Using-persis_in-field
