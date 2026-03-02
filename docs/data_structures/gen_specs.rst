.. _datastruct-gen-specs:

Generator Specs
===============

Used to specify the generator, its inputs and outputs, and user data.

.. code-block:: python
    :linenos:

    ...
    import numpy as np
    from libensemble import GenSpecs
    from generator import gen_random_sample

    ...

    gen_specs = GenSpecs(
        gen_f=gen_random_sample,
        outputs=[("x", float, (1,))],
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

.. note::

  * In all interfaces, custom fields should only be placed in ``"user"``
  * Generator ``"out"`` fields typically match Simulation ``"in"`` fields, and vice-versa.

.. _more example uses: https://github.com/Libensemble/libensemble/wiki/Using-persis_in-field
.. _sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/gen_funcs/sampling.py
.. _test_persistent_aposmm_nlopt.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_nlopt.py
.. _test_persistent_aposmm_with_grad.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py
.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py
