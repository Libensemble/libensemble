.. image:: docs/images/libE_logo.png
   :align: center
   :alt: libEnsemble

|

.. image:: https://img.shields.io/pypi/v/libensemble.svg?color=blue
   :target: https://pypi.org/project/libensemble

.. image:: https://travis-ci.org/Libensemble/libensemble.svg?branch=master
   :target: https://travis-ci.org/Libensemble/libensemble

.. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge/?maxAge=2592000/?branch=master
   :target: https://coveralls.io/github/Libensemble/libensemble?branch=master

.. image:: https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status

|
.. after_badges_rst_tag

===========================
Introduction to libEnsemble
===========================

libEnsemble is a Python library to coordinate the concurrent evaluation of
dynamic ensembles of calculations. The library is developed to use massively
parallel resources to accelerate the solution of design, decision, and
inference problems and to expand the class of problems that can benefit from
increased concurrency levels.

libEnsemble aims for:

• Extreme scaling
• Resilience/fault tolerance
• Monitoring/killing jobs (and recovering resources)
• Portability and flexibility
• Exploitation of persistent data/control flow.

The user selects or supplies a function that generates simulation
input as well as a function that performs and monitors the
simulations. For example, the generation function may contain an
optimization routine to generate new simulation parameters on-the-fly based on the
results of previous simulations. Examples and templates of such functions are
included in the library.

libEnsemble employs a manager-worker scheme that can run on various
communication media (including MPI, multiprocessing, and TCP); interfacing with
user-provided executables is also supported. Each worker can
control and monitor any level of work from small sub-node jobs to huge
many-node simulations. A job controller interface is provided to ensure scripts
are portable, resilient and flexible; it also enables automatic detection of
the nodes and cores in a system and can split up jobs automatically if resource
data isn't supplied.

.. only:: html

  .. include:: deps.rst

