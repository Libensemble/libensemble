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

========================================================
libEnsemble: A complete toolkit for dynamic ensembles of calculations
========================================================

Construct *adaptive*, *portable*, and *scalable* software connecting "deciders" to experiments or simulations.

• **Adaptive ensembles**: Generate parallel tasks *on-the-fly* based on previous computations.
• **Extreme portability and scaling**: Run on or across **laptops**, **clusters**, and **leadership-class machines**.
• **Dynamic resource management**: Adaptively assign and reassign resources (including **GPUs**) to tasks.
• **Application monitoring**: Ensemble members can **run, monitor, and cancel apps**.
• **Coordinated data-flow between tasks**: libEnsemble can pass data between stateful ensemble members.
• **Low start-up cost**: No additional background services or processes required.

libEnsemble is especially effective at solving design, decision, and inference problems on parallel resources.

`Online Documentation`_

.. before_dependencies_rst_tag

Installation
============

Install libEnsemble and its dependencies from PyPI_ using pip::

    pip install libensemble

Other install methods are described in the docs_.

Resources
=========

**Support:**

- Open issues or ask questions on GitHub_.
- Email questions or request `libEnsemble Slack page`_ access from ``libEnsemble@lists.mcs.anl.gov``.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- Contributions_ to libEnsemble are welcome.
- Production functions and workflows can be browsed in the `Community Examples repository`_.

**Citation:**

- Please use the following to cite libEnsemble:

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
.. _Online Documentation: https://libensemble.readthedocs.io/
.. _GitHub: https://github.com/Libensemble/libensemble
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _MPICH: http://www.mpich.org/
.. _mpmath: http://mpmath.org/
.. _PyPI: https://pypi.org
.. _ReadtheDocs: http://libensemble.readthedocs.org/
