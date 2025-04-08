Example Scheduler Submission Scripts
====================================

Below are example submission scripts used to configure and launch libEnsemble
on a variety of high-powered systems. See :ref:`Running on HPC Systems<platform-index>`
for more information about the respective systems and configuration.

.. note::
    It is **highly recommended** that the directive lines (e.g., #SBATCH) in batch
    submission scripts do **NOT** specify processor, task, or GPU configuration
    information---these lines should only specify the number of nodes required.

    For example, do not specify ``#SBATCH --gpus-per-node=4`` in order to use four
    GPUs on the node, when each worker may use less than this, as this may assign
    all of the GPUs to a single MPI invocation. Instead, the configuration should
    be supplied either
    :doc:`in the simulation function<../examples/sim_funcs/forces_simf_gpu>`
    or, if using dynamic resources,
    :doc:`in the generator<../examples/sim_funcs/forces_simf_gpu_vary_resources>`.

General examples
----------------

Slurm - Basic
^^^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_slurm_simple.sh
    :caption: /examples/libE_submission_scripts/submit_slurm_simple.sh
    :language: bash

PBS - Basic
^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_simple.sh
    :caption: /examples/libE_submission_scripts/submit_pbs_simple.sh
    :language: bash

LSF - Basic
^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_lsf_simple.sh
    :caption: /examples/libE_submission_scripts/submit_lsf_simple.sh
    :language: bash

System Examples
---------------

Aurora
^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_aurora.sh
    :caption: /examples/libE_submission_scripts/submit_pbs_aurora.sh
    :language: bash

Frontier (Large WarpX Ensemble)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_frontier_large.sh
    :caption: /examples/libE_submission_scripts/submit_frontier_large.sh
    :language: bash

Perlmutter
^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_perlmutter.sh
    :caption: /examples/libE_submission_scripts/submit_perlmutter.sh
    :language: bash

Polaris
^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_polaris.sh
    :caption: /examples/libE_submission_scripts/submit_pbs_polaris.sh
    :language: bash

Bebop
^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_pbs_central.sh
    :caption: /examples/libE_submission_scripts/bebop_submit_pbs_central.sh
    :language: bash

Bridges - MPI / Central Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/bridges_submit_slurm_central.sh
    :caption: /examples/libE_submission_scripts/bridges_submit_slurm_central.sh
    :language: bash

.. _slurm_mpi_distributed:

SLURM - MPI / Distributed Mode (co-locate workers & MPI applications)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/submit_distrib_mpi4py.sh
    :caption: /examples/libE_submission_scripts/submit_distrib_mpi4py.sh
    :language: bash

Summit (Decommissioned) - On Launch Nodes with Multiprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  literalinclude:: ../../examples/libE_submission_scripts/summit_submit_mproc.sh
    :caption: /examples/libE_submission_scripts/summit_submit_mproc.sh
    :language: bash
