.. _rel-conda:

Conda release
=============

libEnsemble is released as part of the `conda-forge`_ distribution.
This enables users to ``conda install`` the package.

The Conda package is created from the `conda-forge feedstock repository`_.
This repository comes with some common dependencies and automatically creates
three variants (no-mpi, mpich, Open MPI).

Automatic PR
------------

Note that once libEnsemble has been released on PYPI a conda-forge bot will
usually detect the new release and automatically create a pull request with the
changes below. It may take a few hours for this to happen. If no other changes
are required (e.g. new dependencies), then you can simply wait for the tests to
pass and merge.

Manual PR
---------

If necessary, a manual PR can be created as follows.

Create a fork of the repository (not a branch). In the file ``recipe/meta.yaml``
bump the ``version number``, set the ``build number`` to zero, and update the
``sha256``. The latter can be obtained by running ``sha256sum`` on the github
tarball. E.g.~ For v0.6.0::

    sha256sum libensemble-0.6.0.tar.gz

Then, use the phrase `@conda-forge-admin, please rerender` in a comment in
the pull request for automated rerendering. The github-actions bot will
reply with a message when ready to merge.

Release
-------

Approvals from other libEnsemble administrators will be required.
Once the pull request is merged, the new package should become available to
Conda, in the `conda-forge` channel, after a processing delay.

You can then check the three versions:

* `conda install libensemble`
* `conda install libensemble=*=mpi_mpich*`.
* `conda install libensemble=*=mpi_openmpi*`

.. _conda-forge feedstock repository: https://github.com/conda-forge/libensemble-feedstock
.. _conda-forge: https://conda-forge.org/
