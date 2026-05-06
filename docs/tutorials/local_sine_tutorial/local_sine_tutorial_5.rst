5. Next steps
=============

`Introduction <local_sine_tutorial.html>`__ \|\| `1. Getting started <local_sine_tutorial_1.html>`__ \|\| `2. Generator <local_sine_tutorial_2.html>`__ \|\| `3. Simulator <local_sine_tutorial_3.html>`__ \|\| `4. Script <local_sine_tutorial_4.html>`__ \|\| **5. Next steps**

**libEnsemble with MPI**

MPI_ is a standard interface for parallel computing, implemented in libraries
such as MPICH_ and used at extreme scales. MPI potentially allows libEnsemble's
processes to be distributed over multiple nodes and works in some
circumstances where Python's multiprocessing does not. In this section, we'll
explore modifying the above code to use MPI instead of multiprocessing.

We recommend the MPI distribution MPICH_ for this tutorial, which can be found
for a variety of systems here_. You also need mpi4py_, which can be installed
with ``pip install mpi4py``. If you'd like to use a specific version or
distribution of MPI instead of MPICH, configure mpi4py with that MPI at
installation with ``MPICC=<path/to/MPI_C_compiler> pip install mpi4py`` If this
doesn't work, try appending ``--user`` to the end of the command. See the
mpi4py_ docs for more information.

Verify that MPI has been installed correctly with ``mpirun --version``.

**Modifying the script**

Only a few changes are necessary to make our code MPI-compatible. For starters,
comment out the ``libE_specs`` definition:

.. literalinclude:: ../../../libensemble/tests/functionality_tests/test_local_sine_tutorial_3.py
    :language: python
    :start-at: # libE_specs = LibeSpecs
    :end-at: # libE_specs = LibeSpecs

We'll be parameterizing our MPI runtime with a ``parse_args=True`` argument to
the ``Ensemble`` class instead of ``libE_specs``. We'll also use an ``ensemble.is_manager``
attribute so only the first MPI rank runs the data-processing code.

The bottom of your calling script should now resemble:

.. literalinclude:: ../../../libensemble/tests/functionality_tests/test_local_sine_tutorial_3.py
    :linenos:
    :lineno-start: 28
    :language: python
    :start-at: # replace libE_specs

With these changes in place, our libEnsemble code can be run with MPI by

.. code-block:: bash

    mpirun -n 5 python calling.py

where ``-n 5`` tells ``mpirun`` to produce five processes, one of which will be
the manager process with the libEnsemble manager and the other four will run
libEnsemble workers.

This tutorial is only a tiny demonstration of the parallelism capabilities of
libEnsemble. libEnsemble has been developed primarily to support research on
High-Performance computers, with potentially hundreds of workers performing
calculations simultaneously. Please read our
:doc:`platform guides <../../platforms/platforms_index>` for introductions to using
libEnsemble on many such machines.

libEnsemble's Executors can launch non-Python user applications and simulations across
allocated compute resources. Try out this feature with a more-complicated
libEnsemble use-case within our
:doc:`Electrostatic Forces tutorial <../executor_forces_tutorial>`.

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface
.. _MPICH: https://www.mpich.org/
.. _here: https://www.mpich.org/downloads/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/install.html
