.. _rel-conda:

Conda release
=============

libEnsemble is released as part of the `conda-forge`_ distribution.
This enables users to ``conda install`` the package.

The Conda package is created from the `conda-forge feedstock repository`_.
This repository comes with some common dependencies and automatically creates
three variants (no-mpi, mpich, Open MPI).

Create a fork of the repository (not a branch). In the file ``recipe/meta.yaml``
bump the ``version number``, set the ``build number`` to zero, and update the
``sha256``. The latter can be obtained by running ``sha256sum`` on the github
tarball. E.g.~ For v0.6.0::

    sha256sum libensemble-0.6.0.tar.gz

Then, use the phrase `@conda-forge-admin, please rerender` in a comment in
the pull request for automated rerendering. The github-actions bot will
reply with a message when ready to merge.

Approvals from other libEnsemble administrators will be required.

Once the pull request is merged, the new package should become available to
Conda, in the `conda-forge` channel, after a processing delay.

You can then check the three versions:

* `conda install libensemble`
* `conda install libensemble=*=mpi_mpich*`.
* `conda install libensemble=*=mpi_openmpi*`

.. _conda-forge feedstock repository: https://github.com/conda-forge/libensemble-feedstock
.. _conda-forge: https://conda-forge.org/
