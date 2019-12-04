Example Job Submission Scripts
==============================

Below are some example job-submission scripts used to configure and launch libEnsemble
on a variety of high-powered systems. See :doc:`here<platforms_index>` for more
information about the respective systems and configuration.

Bebop - Central mode
--------------------

..  literalinclude:: ../../examples/job_submission_scripts/bebop_submit_slurm_central.sh
    :caption: /examples/job_submission_scripts/bebop_submit_slurm_central.sh
    :language: bash

Bebop - Distributed mode
------------------------

..  literalinclude:: ../../examples/job_submission_scripts/bebop_submit_slurm_distrib.sh
    :caption: /examples/job_submission_scripts/bebop_submit_slurm_distrib.sh
    :language: bash

Cori - Central mode
-------------------

..  literalinclude:: ../../examples/job_submission_scripts/cori_submit.sh
    :caption: /examples/job_submission_scripts/cori_submit.sh
    :language: bash

Blues (Blue Gene Q) - Distributed mode
--------------------------------------

..  literalinclude:: ../../examples/job_submission_scripts/blues_script.pbs
    :caption: /examples/job_submission_scripts/blues_script.pbs
    :language: bash

Theta - On MOM node with multiprocessing
----------------------------------------

..  literalinclude:: ../../examples/job_submission_scripts/theta_submit_mproc.sh
    :caption: /examples/job_submission_scripts/theta_submit_mproc.sh
    :language: bash

Theta - Central mode with Balsam
--------------------------------

..  literalinclude:: ../../examples/job_submission_scripts/theta_submit_balsam.sh
    :caption: /examples/job_submission_scripts/theta_submit_balsam.sh
    :language: bash

Summit - On launch nodes with multiprocessing
---------------------------------------------

..  literalinclude:: ../../examples/job_submission_scripts/summit_submit_mproc.sh
    :caption: /examples/job_submission_scripts/summit_submit_mproc.sh
    :language: bash
