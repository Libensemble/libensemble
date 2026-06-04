pixi
====

`Advanced Installation <advanced_installation.html>`__ \|\| `pip <advanced_installation_pip.html>`__ \|\| `uv <advanced_installation_uv.html>`__ \|\| **pixi** \|\| `conda <advanced_installation_conda.html>`__ \|\| `Spack <advanced_installation_spack.html>`__

Add to your pixi_ environment::

    pixi add libensemble

libEnsemble is also distributed with locked pixi environments for different versions of Python
and various dependency sets, primarily for testing but also useful for guaranteed working environments.
See a list with::

    pixi workspace environment list

and activate with::

    pixi shell -e <environment_name>

.. _pixi: https://pixi.prefix.dev/latest/
