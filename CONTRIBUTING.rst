Contributing to libEnsemble
===========================

Contributions may be made via a GitHub pull request to

    https://github.com/Libensemble/libensemble

libEnsemble uses the Gitflow model. Contributors should branch from, and
make pull requests to, the develop branch. The main branch is used only
for releases. Pull requests may be made from a fork, for those without
repository write access.

Code should pass flake8 tests, allowing for the exceptions
given in the flake8_ file in the project directory.
Python code should be formatted using the latest version of black_ by running
the following in the base libensemble directory::

  black --config=.black .

Issues can be raised at

    https://github.com/Libensemble/libensemble/issues

Issues may include reporting bugs or suggested features. Administrators
will add issues, as appropriate, to the project board at

    https://github.com/Libensemble/libensemble/projects

By convention, user branch names should have a <type>/<name> format, where
example types are feature, bugfix, testing, docs, and experimental.
Administrators may take a hotfix branch from the main, which will be
merged into main (as a patch) and develop. Administrators may also take a
release branch off develop and then merge this branch into main and develop
for a release. Most branches should relate to an issue or feature.

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number. This will automatically
close out the issues when they are pulled into the default branch (currently
main).

libEnsemble is distributed under a 3-clause BSD license (see LICENSE).  The
act of submitting a pull request (with or without an explicit
Signed-off-by tag) will be understood as an affirmation of the
following:

::

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.

.. _black: https://pypi.org/project/black/
.. _flake8: https://github.com/Libensemble/libensemble/blob/develop/.flake8
