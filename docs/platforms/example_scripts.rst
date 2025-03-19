Example Scheduler Submission Scripts
====================================

Below are example submission scripts used to configure and launch libEnsemble
on a variety of high-powered systems. See :ref:`here<platform-index>` for more
information about the respective systems and configuration.

General examples
----------------

.. dropdown:: Slurm - Basic

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_slurm_simple.sh
        :caption: /examples/libE_submission_scripts/submit_slurm_simple.sh
        :language: bash

.. dropdown:: PBS - Basic

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_simple.sh
        :caption: /examples/libE_submission_scripts/submit_pbs_simple.sh
        :language: bash

.. dropdown:: LSF - Basic

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_lsf_simple.sh
        :caption: /examples/libE_submission_scripts/submit_lsf_simple.sh
        :language: bash


System Examples
---------------

.. dropdown:: Aurora

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_aurora.sh
        :caption: /examples/libE_submission_scripts/submit_pbs_aurora.sh
        :language: bash

.. dropdown:: Frontier (Large WarpX Ensemble)

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_frontier_large.sh
        :caption: /examples/libE_submission_scripts/submit_frontier_large.sh
        :language: bash


.. dropdown:: Perlmutter

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_perlmutter.sh
        :caption: /examples/libE_submission_scripts/submit_perlmutter.sh
        :language: bash

.. dropdown:: Polaris

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_pbs_polaris.sh
        :caption: /examples/libE_submission_scripts/submit_pbs_polaris.sh
        :language: bash

.. dropdown:: Bridges - Central Mode

    ..  literalinclude:: ../../examples/libE_submission_scripts/bridges_submit_slurm_central.sh
        :caption: /examples/libE_submission_scripts/bridges_submit_slurm_central.sh
        :language: bash

.. dropdown:: Bebop - Central Mode

    ..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_slurm_central.sh
        :caption: /examples/libE_submission_scripts/bebop_submit_slurm_central.sh
        :language: bash

.. dropdown:: Bebop - Distributed Mode

    ..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_pbs_distrib.sh
        :caption: /examples/libE_submission_scripts/bebop_submit_pbs_distrib.sh
        :language: bash

.. dropdown:: Summit (Decommissioned) - On Launch Nodes with Multiprocessing

    ..  literalinclude:: ../../examples/libE_submission_scripts/summit_submit_mproc.sh
        :caption: /examples/libE_submission_scripts/summit_submit_mproc.sh
        :language: bash
