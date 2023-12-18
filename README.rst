.. image:: https://raw.githubusercontent.com/Libensemble/libensemble/main/docs/images/libE_logo.png
   :align: center
   :alt: libEnsemble

|

.. image:: https://img.shields.io/pypi/v/libensemble.svg?color=blue
   :target: https://pypi.org/project/libensemble

.. image:: https://img.shields.io/conda/v/conda-forge/libensemble?color=blue
   :target: https://anaconda.org/conda-forge/libensemble

.. image:: https://img.shields.io/spack/v/py-libensemble?color=blue
   :target: https://packages.spack.io/package.html?name=py-libensemble

|

.. image:: https://github.com/Libensemble/libensemble/actions/workflows/extra.yml/badge.svg?branch=main
   :target: https://github.com/Libensemble/libensemble/actions

.. image:: https://codecov.io/github/Libensemble/libensemble/graph/badge.svg
   :target: https://codecov.io/github/Libensemble/libensemble

.. image:: https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://joss.theoj.org/papers/10.21105/joss.06031/status.svg
   :target: https://doi.org/10.21105/joss.06031
   :alt: JOSS Status

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

Basic Usage
===========

Create an ``Ensemble``, then customize it with general settings, simulation and generator parameters,
and an exit condition. Run the following via ``python this_file.py --comms local --nworkers 4``:

.. code-block:: python

   import numpy as np

   from libensemble import Ensemble
   from libensemble.gen_funcs.sampling import uniform_random_sample
   from libensemble.sim_funcs.six_hump_camel import six_hump_camel
   from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
   from libensemble.tools import add_unique_random_streams

   if __name__ == "__main__":
       sampling = Ensemble(parse_args=True)
       sampling.sim_specs = SimSpecs(
           sim_f=six_hump_camel,
           inputs=["x"],
           outputs=[("f", float)],
       )
       sampling.gen_specs = GenSpecs(
           gen_f=uniform_random_sample,
           outputs=[("x", float, (2,))],
           user={
               "gen_batch_size": 500,
               "lb": np.array([-3, -2]),
               "ub": np.array([3, 2]),
           },
       )

       sampling.persis_info = add_unique_random_streams({}, sampling.nworkers + 1)
       sampling.exit_criteria = ExitCriteria(sim_max=101)
       sampling.run()
       sampling.save_output(__file__)

       if sampling.is_manager:
           print("Some output data:\n", sampling.H[["x", "f"]][:10])

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
    author  = {Stephen Hudson and Jeffrey Larson and John-Luke Navarro and Stefan M. Wild},
    journal = {{IEEE} Transactions on Parallel and Distributed Systems},
    volume  = {33},
    number  = {4},
    pages   = {977--988},
    year    = {2022},
    doi     = {10.1109/tpds.2021.3082815}
  }

.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples
.. _conda-forge: https://conda-forge.org/
.. _Contributions: https://github.com/Libensemble/libensemble/blob/main/CONTRIBUTING.rst
.. _docs: https://libensemble.readthedocs.io/en/main/advanced_installation.html
.. _GitHub: https://github.com/Libensemble/libensemble
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _MPICH: http://www.mpich.org/
.. _mpmath: http://mpmath.org/
.. _PyPI: https://pypi.org
.. _Quickstart: https://libensemble.readthedocs.io/en/main/introduction.html
.. _ReadtheDocs: http://libensemble.readthedocs.org/
