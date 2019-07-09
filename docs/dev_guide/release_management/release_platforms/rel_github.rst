.. _rel-github:

Github release
==============

The administrator should follow the github instructions to draft a new release. These can currently be found at: https://help.github.com/en/articles/creating-releases

Both the version and title will be of the form vX.Y.Z::

    E.g. v0.5.0.

From version 1.0, these should follow semantic versioning where, where X/Y/Z are major, minor and patch revisions.

Prior to version 1.0, the second number may include breaking API changes, and the third number may include minor additions.

The release notes should be included in the description. These should already be in `docs/release_notes.rst`. The release nots should be copied just the current release, starting from the date. Hint: To see example of raw input click *edit* next to one of the previous releases.

Note, unlike some platforms (e.g. PyPI), github releases can be edited or deleted once created.
