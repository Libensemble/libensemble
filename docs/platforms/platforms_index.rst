
libEnsemble has been largely developed, supported, and tested on Linux
distributions and macOS, from laptops to thousands of compute-nodes. Although
libEnsemble and most user functions are cross-platform compatible, there are
platform-specific differences for installing and configuring libEnsemble.

Personal Computers
==================

Users interested in installing and running libEnsemble on their personal machines
are encouraged to start by reading the Quickstart guide :doc:`here<../quickstart>`.

We recommend installing libEnsemble and it's dependencies in a virtual environment,
created either through ``conda create``, ``virtualenv``, or ``python -m venv``,
depending on how Python is installed.


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

Read more about configuring and launching libEnsemble on some HPC systems:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    bebop
    theta
    summit
    example_scripts


.. _Balsam: https://balsam.readthedocs.io/en/latest/
