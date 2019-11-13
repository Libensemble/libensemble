HPC Systems
===========

libEnsemble's flexible architecture lends it best to two general modes of worker
distributions across allocated compute nodes. The first mode we refer
to as *centralized* mode, where the libEnsemble manager and worker processes
are grouped on one or more nodes, but through the libEnsemble job-controller or a
job-launch command can execute calculations on the other allocated nodes:

.. image:: ../images/centralized_Bb.png
    :alt: centralized
    :scale: 75
    :align: center

Alternatively, in *distributed* mode, each worker process runs independently of
other workers directly on one or more allocated nodes:

.. image:: ../images/distributed_Bb.png
    :alt: distributed
    :scale: 75
    :align: center

.. note::

    Certain machines (like Theta and Summit) that can only submit MPI jobs from
    specialized launch nodes do not support libEnsemble in distributed mode.

Due to this factor, libEnsemble on Theta and Summit approaches centralized mode
differently. On these machines, libEnsemble is run centralized on either a
compute-node with the support of Balsam_ or on a frontend server called a MOM
(Machine-Oriented Mini-server) node.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
