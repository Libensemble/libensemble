.. _rel-pypi:

PyPI release
============

libEnsemble is released on the Python Package Index (commonly known as PyPI).
This enables users to "``pip install``" the package.

The package is stored on PyPI in the form of a source distribution (commonly
known as a tarball). The tarball could be obtained from GitHub, though
historically this has been created with a checkout of libEnsemble from git.

You will need logon credentials for the libEnsemble PyPI. You will also need
twine (which can be pip or conda installed).

In the package directory on the master branch (the one containing setup.py) do
the following:

Create distribution::

    python setup.py sdist

Upload (you will need username/password here)::

    twine upload dist/*

If you now do "``pip install libensemble``" it should find the new version.

It should also be visible here:

https://pypi.org/project/libensemble/

For more details on creating PyPI packages see:

https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
