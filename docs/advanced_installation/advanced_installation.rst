Advanced Installation
=====================

`pip <advanced_installation_pip.html>`__ \|\| `uv <advanced_installation_uv.html>`__ \|\| `pixi <advanced_installation_pixi.html>`__ \|\| `conda <advanced_installation_conda.html>`__ \|\| `Spack <advanced_installation_spack.html>`__

libEnsemble can be installed from ``pip``, ``uv``, ``pixi``, ``Conda``, or ``Spack``.

libEnsemble requires the following dependencies, which are typically
automatically installed alongside libEnsemble:

* Python_       ``>= 3.11``
* NumPy_        ``>= 1.21``
* psutil_       ``>= 5.9.4``
* `pydantic`_   ``>= 2``
* gest-api_     ``>= 0.1,<0.2``

We recommend installing in a virtual environment from ``uv``, ``conda`` or another source.

Further recommendations for selected HPC systems are given in the
:ref:`HPC platform guides<platform-index>`.

.. toctree::
    :hidden:

    advanced_installation_pip
    advanced_installation_uv
    advanced_installation_pixi
    advanced_installation_conda
    advanced_installation_spack

Globus Compute
--------------

`Globus Compute`_ may be installed optionally to submit simulation function instances to remote Globus Compute endpoints.

.. _Globus Compute: https://www.globus.org/compute
.. _Python: http://www.python.org
.. _NumPy: http://www.numpy.org
.. _psutil: https://pypi.org/project/psutil/
.. _pydantic: https://docs.pydantic.dev/1.10/
.. _gest-api: https://github.com/campa-consortium/gest-api
