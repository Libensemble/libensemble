Release Process
===============

A release can be undertaken only by a project administrator. A project
administrator should have an administrator role on the libEnsemble GitHub,
PyPI, and readthedocs pages.

Before release
--------------

- A GitHub issue is created with a checklist for the release.

- A release branch should be taken off develop (or develop pulls controlled).

- Release notes for this version are added to the documentation with release
  date, including a list of supported (tested) platforms.

- Version number is updated wherever it appears (and ``+dev`` suffix is removed)
  (in ``libensemble/version.py``).

- Year in ``LICENSE`` is checked for correctness.
  (Note: The year generated in docs by ``docs/conf.py`` should be automatic).

- ``pyproject.toml`` and ``libensemble/__init__.py`` are checked to ensure all
  information is up to date.

- Update ``.wci.yml`` in root directory (version, date and any other
  information).

- ``MANIFEST.in`` is checked. Locally, try out ``python setup.py sdist`` and
  check the created tarball contains correct files and directories for PyPI
  package.

- Tests are run with source to be released (this may iterate):

  - On-line CI (GitHub Actions) tests must pass.

  - Launch and verify extra tests:
    ``gh workflow run libEnsemble-complete-CI --ref branch_name``

  - Scaling tests must be run on HPC platforms listed as supported in release
    notes. Test variants by platform, launch mechanism, scale, and other
    factors can be configured and exported by the libE-Templater_.

  - Coverage must not have decreased unless there is a justifiable reason.

  - Documentation must build and display correctly wherever hosted (currently
    readthedocs.com).

- Locally check that the example code in the README works with both local
  comms and mpi4py.

- Pull request from either the develop or release branch to main requesting
  one or more reviewers (including at least one other administrator).

- Reviewer will check that all tests have passed and will then approve merge.

During release
--------------

An administrator will take the following steps.

- Merge the pull request into main.

- Make the following releases:

  - A GitHub release will be taken from the main
    (:ref:`github release<rel-github>`).

  - A tarball (source distribution) will be uploaded to PyPI
    (:ref:`PyPI release<rel-pypi>`).

  - The Conda package will be updated (:ref:`Conda release<rel-conda>`).

  - Spack package will be updated (:ref:`Spack release<rel-spack>`).

- If the merge was made from a release branch (instead of develop), merge this
  branch into develop.

- Create a new commit on develop that appends ``+dev`` to the version number
  (wherever is appears).

After release
-------------

- Email the libEnsemble mailing list, and notify the `everyone` channel in the
  libEnsemble Slack workspace with an announcement and quick summary of the
  release.

- Ensure all relevant GitHub issues are closed and moved to the *Done* column
  on the kanban project board (inc. the release checklist). Those that were
  already in *Done* should be archived.

.. _libE-Templater: https://github.com/Libensemble/libE-templater
