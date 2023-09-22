.. _datastruct-platform-specs:

Platform Specs
==============

libEnsemble detects platform specifications including MPI runners and resources.
Usually this will result in the correct settings. However, users can configure
platform specifications via the `platform_specs`_ option or indicate a known
platform via the `platform`_ option.

platform_specs
--------------

A ``Platform`` object or dictionary specifying settings for a platform.

To define a platform (in calling script):

.. tab-set::

    .. tab-item:: Platform Object

       .. code-block:: python

           from libensemble.resources.platforms import Platform

           libE_specs["platform_specs"] = Platform(
               mpi_runner="srun",
               cores_per_node=64,
               logical_cores_per_node=128,
               gpus_per_node=8,
               gpu_setting_type="runner_default",
               gpu_env_fallback="ROCR_VISIBLE_DEVICES",
               scheduler_match_slots=False,
           )

    .. tab-item:: Dictionary

       .. code-block:: python

           libE_specs["platform_specs"] = {
               "mpi_runner": "srun",
               "cores_per_node": 64,
               "logical_cores_per_node": 128,
               "gpus_per_node": 8,
               "gpu_setting_type": "runner_default",
               "gpu_env_fallback": "ROCR_VISIBLE_DEVICES",
               "scheduler_match_slots": False,
           }

The list of platform fields is given below. Any fields not given will be
auto-detected by libEnsemble.

.. _platform-fields:

.. dropdown:: ``Platform Fields``
    :open:

    .. autopydantic_model:: libensemble.resources.platforms.Platform
        :model-show-validator-members: False
        :model-show-validator-summary: False
        :field-list-validators: False
        :field-show-default: False
        :member-order:
        :model-show-field-summary: False

To use an existing platform:

.. code-block:: python

    from libensemble.resources.platforms import PerlmutterGPU

    libE_specs["platform_specs"] = PerlmutterGPU()

See :ref:`known platforms<known-platforms>`.

platform
--------

A string giving the name of a known platform defined in the platforms module.

.. code-block:: python

    libE_specs["platform"] = "perlmutter_g"

Note: the environment variable ``LIBE_PLATFORM`` is an alternative way of setting.

E.g., in the command line or batch submission script:

.. code-block:: shell

    export LIBE_PLATFORM="perlmutter_g"

.. _known-platforms:

Known Platforms List
--------------------

.. dropdown:: ``Known_platforms``
    :open:

    .. autopydantic_model:: libensemble.resources.platforms.Known_platforms
        :model-show-validator-members: False
        :model-show-validator-summary: False
        :model-show-field-summary: False
        :field-list-validators: False
        :field-show-required: False
        :field-show-default: False
        :field-show-alias: False
        :member-order:
