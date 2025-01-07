sampling
--------

.. role:: underline
    :class: underline

.. automodule:: sampling
  :members: uniform_random_sample, latin_hypercube_sample
  :undoc-members:

.. dropdown:: :underline:`sampling.py`

   .. literalinclude:: ../../libensemble/gen_funcs/sampling.py
      :language: python
      :linenos:

persistent_sampling
-------------------
.. automodule:: persistent_sampling
  :members: persistent_uniform, persistent_request_shutdown, persistent_uniform_with_cancellations
  :undoc-members:

.. dropdown:: :underline:`persistent_sampling.py`

   .. literalinclude:: ../../libensemble/gen_funcs/persistent_sampling.py
      :language: python
      :linenos:

persistent_sampling_var_resources
---------------------------------
.. automodule:: persistent_sampling_var_resources
  :members: uniform_sample_with_var_gpus
  :undoc-members:
