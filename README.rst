.. image:: docs/images/libEnsemble_Logo.svg
   :align: center
   :alt: libEnsemble

|

.. image:: https://img.shields.io/pypi/v/libensemble.svg?color=blue
   :target: https://pypi.org/project/libensemble

.. image:: https://github.com/Libensemble/libensemble/workflows/libEnsemble-CI/badge.svg?branch=main
   :target: https://github.com/Libensemble/libensemble/actions

.. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge.svg?branch=main
   :target: https://coveralls.io/github/Libensemble/libensemble?branch=main

.. image:: https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

|

.. after_badges_rst_tag

=====================================================================
libEnsemble: A complete toolkit for dynamic ensembles of calculations
=====================================================================

Adaptive, portable, and scalable software for connecting "deciders" to experiments or simulations.

• **Dynamic ensembles**: Generate parallel tasks on-the-fly based on previous computations.
• **Extreme portability and scaling**: Run on or across laptops, clusters, and leadership-class machines.
• **Heterogeneous computing**: Dynamically and portably assign CPUs, GPUs, or multiple nodes.
• **Application monitoring**: Ensemble members can run, monitor, and cancel apps.
• **Data-flow between tasks**: Running ensemble members can send and receive data.
• **Low start-up cost**: No additional background services or processes required.

libEnsemble is effective at solving design, decision, and inference problems on parallel resources.

`Quickstart`_

Installation
============

Install libEnsemble and its dependencies from PyPI_ using pip::

    pip install libensemble

Other install methods are described in the docs_.

Resources
=========

**Support:**

- Ask questions or report issues on GitHub_.
- Email ``libEnsemble@lists.mcs.anl.gov`` to request `libEnsemble Slack page`_.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- Contributions_ to libEnsemble are welcome.
- Browse production functions and workflows in the `Community Examples repository`_.

**Cite libEnsemble:**

.. code-block:: bibtex

  @article{Hudson2022,
    title   = {{libEnsemble}: A Library to Coordinate the Concurrent
               Evaluation of Dynamic Ensembles of Calculations},
    author  = {Stephen Hudson and Jeffrey Larson and John-Luke Navarro and Stefan Wild},
    journal = {{IEEE} Transactions on Parallel and Distributed Systems},
    volume  = {33},
    number  = {4},
    pages   = {977--988},
    year    = {2022},
    doi     = {10.1109/tpds.2021.3082815}
  }

.. _conda-forge: https://conda-forge.org/
.. _Contributions: https://github.com/Libensemble/libensemble/blob/main/CONTRIBUTING.rst
.. _docs: https://libensemble.readthedocs.io/en/main/advanced_installation.html
.. _GitHub: https://github.com/Libensemble/libensemble
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _MPICH: http://www.mpich.org/
.. _mpmath: http://mpmath.org/
.. _Quickstart: https://libensemble.readthedocs.io/en/main/introduction.html
.. _PyPI: https://pypi.org
.. _ReadtheDocs: http://libensemble.readthedocs.org/
