Example libEnsemble Submission Scripts
======================================

Below are example submission scripts used to configure and launch libEnsemble
on a variety of high-powered systems. See :doc:`here<platforms_index>` for more
information about the respective systems and configuration.

Alternatively to interacting with the scheduler or configuring submission scripts,
libEnsemble now features a portable set of :ref:`command-line utilities<liberegister>`
for submitting workflows to almost any system or scheduler.

Slurm - Basic
-------------

..  literalinclude:: ../../examples/libE_submission_scripts/submit_slurm_simple.sh
    :caption: /examples/libE_submission_scripts/submit_slurm_simple.sh
    :language: bash

Bridges - Central Mode
----------------------

..  literalinclude:: ../../examples/libE_submission_scripts/bridges_submit_slurm_central.sh
    :caption: /examples/libE_submission_scripts/bridges_submit_slurm_central.sh
    :language: bash

Bebop - Central Mode
--------------------

..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_slurm_central.sh
    :caption: /examples/libE_submission_scripts/bebop_submit_slurm_central.sh
    :language: bash

Bebop - Distributed Mode
------------------------

..  literalinclude:: ../../examples/libE_submission_scripts/bebop_submit_slurm_distrib.sh
    :caption: /examples/libE_submission_scripts/bebop_submit_slurm_distrib.sh
    :language: bash

Cori - Central Mode
-------------------

..  literalinclude:: ../../examples/libE_submission_scripts/cori_submit.sh
    :caption: /examples/libE_submission_scripts/cori_submit.sh
    :language: bash

Blues (Blue Gene Q) - Distributed Mode
--------------------------------------

..  literalinclude:: ../../examples/libE_submission_scripts/blues_script.pbs
    :caption: /examples/libE_submission_scripts/blues_script.pbs
    :language: bash

Theta - On MOM Node with Multiprocessing
----------------------------------------

..  literalinclude:: ../../examples/libE_submission_scripts/theta_submit_mproc.sh
    :caption: /examples/libE_submission_scripts/theta_submit_mproc.sh
    :language: bash

Theta - Central Mode with Balsam
--------------------------------

..  literalinclude:: ../../examples/libE_submission_scripts/theta_submit_balsam.sh
    :caption: /examples/libE_submission_scripts/theta_submit_balsam.sh
    :language: bash

Summit - On Launch Nodes with Multiprocessing
---------------------------------------------

..  literalinclude:: ../../examples/libE_submission_scripts/summit_submit_mproc.sh
    :caption: /examples/libE_submission_scripts/summit_submit_mproc.sh
    :language: bash
