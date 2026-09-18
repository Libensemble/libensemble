pixi
====

:doc:`Advanced Installation <advanced_installation>` || :doc:`pip <advanced_installation_pip>` || :doc:`uv <advanced_installation_uv>` || **pixi** || :doc:`conda <advanced_installation_conda>` || :doc:`Spack <advanced_installation_spack>`

Add to your pixi_ environment::

    pixi add libensemble

libEnsemble is also distributed with locked pixi environments for different versions of Python
and various dependency sets, primarily for testing but also useful for guaranteed working environments.
See a list with::

    pixi workspace environment list

and activate with::

    pixi shell -e <environment_name>

.. _pixi: https://pixi.prefix.dev/latest/
