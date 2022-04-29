.. _rel-pypi:

PyPI release
============

libEnsemble is released on the Python Package Index (commonly known as PyPI).
This enables users to ``pip install`` the package.

The package is stored on PyPI in the form of a source distribution (commonly
known as a tarball). The tarball should be created as detailed below (which
creates the distribution package using the MANIFEST.in file in the git root
directory. Do not use the tarball on GitHub, which does not follow MANIFEST.in
and does not contain the required PKG-INFO file.

You will need logon credentials for the libEnsemble PyPI. You will also need
twine (which can be pip or Conda installed).

In the package directory on the main branch (the one containing setup.py) do
the following:

Create distribution::

    python setup.py sdist

Upload (you will need username/password here)::

    twine upload dist/*

If you now run ::

    pip install libensemble

it should find the new version.

It should also be visible here:

https://pypi.org/project/libensemble/

For more details on creating PyPI packages see

https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
