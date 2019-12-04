Release Process
===============

This document details the current release process for libEnsemble. A release
can only be undertaken by a project administrator. A project administrator
should have an administrator role on the libEnsemble GitHub, PyPI, and
readthedocs pages.

Before release
--------------

- A GitHub issue is created with a checklist for the release.

- A release branch should be taken off develop (or develop pulls controlled).

- Release notes for this version are added to the documentation with release
  date, including a list of supported (tested) platforms.

- Version number is updated wherever it appears:
  (in ``setup.py``, ``libensemble/__init__.py``, ``README.rst`` and twice in ``docs/conf.py``)

- Check year is correct in ``README.rst`` under *Citing libEnsemble* and in ``docs/conf.py``.

- ``setup.py`` and ``libensemble/__init__.py`` are checked to ensure all information is up to date.

- Tests are run with source to be released (this may iterate):

  - On-line CI (currently Travis) tests must pass.

  - Scaling tests must be run on HPC platforms listed as supported in release notes.

  - Coverage must not have decreased unless there is a justifiable reason.

  - Documentation must build and display correctly wherever hosted (currently readthedocs.com).

- Pull request from either develop or release branch to master requesting
  reviewer/s (including at least one other administrator).

- Reviewer will check tests have passed and approve merge.

During release
--------------

An administrator will take the following steps.

- Merge the pull request into master.

- Once CI tests have passed on master.

  - A GitHub release will be taken from the master (:ref:`github release<rel-github>`).

  - A tarball (source distribution) will be uploaded to PyPI (:ref:`PyPI release<rel-pypi>`).

  - Spack package will be updated (:ref:`Spack release<rel-spack>`).

- If the merge was made from a release branch (instead of develop), merge this branch into develop.

After release
-------------

- Ensure all relevant GitHub issues are closed and moved to the *Done* column
  on the kanban project board (inc. the release checklist).

- Email libEnsemble mailing list
