Example Scheduler Submission Scripts
====================================

Below are example submission scripts used to configure and launch libEnsemble
on a variety of high-powered systems. See :doc:`here<platforms_index>` for more
information about the respective systems and configuration.

Alternatively to interacting with the scheduler or configuring submission scripts,
libEnsemble now features a portable set of :ref:`command-line utilities<liberegister>`
for submitting workflows to almost any system or scheduler.

.. dropdown:: Slurm - Basic

    ..  literalinclude:: ../../examples/libE_submission_scripts/submit_slurm_simple.sh
        :caption: /examples/libE_submission_scripts/submit_slurm_simple.sh
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

    ..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_slurm_distrib.sh
        :caption: /examples/libE_submission_scripts/bebop_submit_slurm_distrib.sh
        :language: bash

.. dropdown:: Summit - On Launch Nodes with Multiprocessing

    ..  literalinclude:: ../../examples/libE_submission_scripts/summit_submit_mproc.sh
        :caption: /examples/libE_submission_scripts/summit_submit_mproc.sh
        :language: bash

.. dropdown:: Cobalt - Intermediate node with Multiprocessing

    .. literalinclude:: ../../examples/libE_submission_scripts/cobalt_submit_mproc.sh
        :caption: /examples/libE_submission_scripts/cobalt_submit_mproc.sh
        :language: bash
