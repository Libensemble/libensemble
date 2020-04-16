Generation Functions
====================

Below are example generation functions available in libEnsemble.

.. IMPORTANT::
  See the API for generation functions :ref:`here<api_gen_f>`.

sampling
--------
.. automodule:: sampling
  :members:
  :undoc-members:

APOSMM
------

Configuring APOSMM
^^^^^^^^^^^^^^^^^^

By default, APOSMM will import several optmizers which require
external packages and MPI. To import only the optmizers you are using
you can add the following lines in your calling script, before importing APOSMM::

    import libensemble.gen_funcs
    libensemble.gen_funcs.rc.aposmm_optimizer = <optimizer>

Where ``optimizer`` can be a string or list of strings.

The options are:

    - ``'petsc'``, ``'nlopt'``, ``'dfols'``, ``'scipy'``, ``'external'``

APOSMM
^^^^^^

.. automodule:: aposmm
  :members:
  :undoc-members:

uniform_or_localopt
-------------------
.. automodule:: uniform_or_localopt
  :members:
  :undoc-members:

persistent_uniform_sampling
---------------------------
.. automodule:: persistent_uniform_sampling
  :members:
  :undoc-members:
