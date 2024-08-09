===================
Simple Introduction
===================

This tutorial demonstrates the capability to perform ensembles of
calculations in parallel using :doc:`libEnsemble<../introduction>`.

We recommend reading this brief :doc:`Overview<../overview_usecases>`.

|Open in Colab|

For this tutorial, our generator will produce uniform randomly sampled
values, and our simulator will calculate the sine of each. By default we don't
need to write a new allocation function.

.. tab-set::

    .. tab-item:: 1. Getting started

        libEnsemble is written entirely in Python_. Let's make sure
        the correct version is installed.

        .. code-block:: bash

            python --version  # This should be >= 3.10

        .. _Python: https://www.python.org/

        For this tutorial, you need NumPy_ and (optionally)
        Matplotlib_ to visualize your results. Install libEnsemble and these other
        libraries with

        .. code-block:: bash

            pip install libensemble
            pip install matplotlib # Optional

        If your system doesn't allow you to perform these installations, try adding
        ``--user`` to the end of each command.

    .. tab-item:: 2. Generator

        Let's begin the coding portion of this tutorial by writing our generator function,
        or :ref:`gen_f<api_gen_f>`.

        An available libEnsemble worker will call this generator function with the
        following parameters:

            * :ref:`InputArray<funcguides-history>`: A selection of the :ref:`History array<funcguides-history>` (*H*),
              passed to the generator function in case the user wants to generate
              new values based on simulation outputs. Since our generator produces random
              numbers, it'll be ignored this time.

            * :ref:`persis_info<datastruct-persis-info>`: Dictionary with worker-specific
              information. In our case, this dictionary contains NumPy Random Stream objects
              for generating random numbers.

            * :ref:`gen_specs<datastruct-gen-specs>`: Dictionary with user-defined static fields and
              parameters. Customizable parameters such as lower and upper bounds and batch
              sizes are placed within the ``gen_specs["user"]`` dictionary.

        Later on, we'll populate :class:`gen_specs<libensemble.specs.GenSpecs>` and ``persis_info`` when we initialize libEnsemble.

        For now, create a new Python file named ``sine_gen.py``. Write the following:

        .. literalinclude:: ../../libensemble/tests/functionality_tests/sine_gen.py
            :language: python
            :linenos:
            :caption: examples/tutorials/simple_sine/sine_gen.py

        Our function creates ``batch_size`` random numbers uniformly distributed
        between the ``lower`` and ``upper`` bounds. A random stream
        from ``persis_info`` is used to generate these values, which are then placed
        into an output NumPy array that matches the dtype from ``gen_specs["out"]``.

    .. tab-item:: 3. Simulator

        Next, we'll write our simulator function or :ref:`sim_f<api_sim_f>`. Simulator
        functions perform calculations based on values from the generator function.
        The only new parameter here is :ref:`sim_specs<datastruct-sim-specs>`, which
        serves a purpose similar to the :class:`gen_specs<libensemble.specs.GenSpecs>` dictionary.

        Create a new Python file named ``sine_sim.py``. Write the following:

        .. literalinclude:: ../../libensemble/tests/functionality_tests/sine_sim.py
            :language: python
            :linenos:
            :caption: examples/tutorials/simple_sine/sine_sim.py

        Our simulator function is called by a worker for every work item produced by
        the generator function. This function calculates the sine of the passed value,
        and then returns it so the worker can store the result.

    .. tab-item:: 4. Script

        Now lets write the script that configures our generator and simulator
        functions and starts libEnsemble.

        Create an empty Python file named ``calling.py``.
        In this file, we'll start by importing NumPy, libEnsemble's setup classes,
        and the generator and simulator functions we just created.

        In a class called :ref:`LibeSpecs<datastruct-libe-specs>` we'll
        specify the number of workers and the manager/worker intercommunication method.
        ``"local"``, refers to Python's multiprocessing.

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial.py
            :language: python
            :linenos:
            :end-at: libE_specs = LibeSpecs

        We configure the settings and specifications for our ``sim_f`` and ``gen_f``
        functions in the :ref:`GenSpecs<datastruct-gen-specs>` and
        :ref:`SimSpecs<datastruct-sim-specs>` classes, which we saw previously
        being passed to our functions *as dictionaries*.
        These classes also describe to libEnsemble what inputs and outputs from those
        functions to expect.

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial.py
            :language: python
            :linenos:
            :lineno-start: 10
            :start-at: gen_specs = GenSpecs
            :end-at: sim_specs_end_tag

        We then specify the circumstances where
        libEnsemble should stop execution in :ref:`ExitCriteria<datastruct-exit-criteria>`.

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial.py
            :language: python
            :linenos:
            :lineno-start: 26
            :start-at: exit_criteria = ExitCriteria
            :end-at: exit_criteria = ExitCriteria

        Now we're ready to write our libEnsemble :doc:`libE<../programming_libE>`
        function call. :ref:`ensemble.H<funcguides-history>` is the final version of
        the history array. ``ensemble.flag`` should be zero if no errors occur.

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial.py
            :language: python
            :linenos:
            :lineno-start: 28
            :start-at: ensemble = Ensemble
            :end-at: print(history)

        That's it! Now that these files are complete, we can run our simulation.

        .. code-block:: bash

            python calling.py

        If everything ran perfectly and you included the above print statements, you
        should get something similar to the following output (although the
        columns might be rearranged).

        .. code-block::

            ["y", "sim_started_time", "gen_worker", "sim_worker", "sim_started", "sim_ended", "x", "allocated", "sim_id", "gen_ended_time"]
            [(-0.37466051, 1.559+09, 2, 2,  True,  True, [-0.38403059],  True,  0, 1.559+09)
            (-0.29279634, 1.559+09, 2, 3,  True,  True, [-2.84444261],  True,  1, 1.559+09)
            ( 0.29358492, 1.559+09, 2, 4,  True,  True, [ 0.29797487],  True,  2, 1.559+09)
            (-0.3783986, 1.559+09, 2, 1,  True,  True, [-0.38806564],  True,  3, 1.559+09)
            (-0.45982062, 1.559+09, 2, 2,  True,  True, [-0.47779319],  True,  4, 1.559+09)
            ...

        In this arrangement, our output values are listed on the far left with the
        generated values being the fourth column from the right.

        Two additional log files should also have been created.
        ``ensemble.log`` contains debugging or informational logging output from
        libEnsemble, while ``libE_stats.txt`` contains a quick summary of all
        calculations performed.

        Here is graphed output using ``Matplotlib``, with entries colored by which
        worker performed the simulation:

        .. image:: ../images/sinex.png
            :alt: sine
            :align: center

        If you want to verify your results through plotting and installed Matplotlib
        earlier, copy and paste the following code into the bottom of your calling
        script and run ``python calling.py`` again

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial.py
            :language: python
            :linenos:
            :lineno-start: 37
            :start-at: import matplotlib
            :end-at: plt.savefig("tutorial_sines.png")

        Each of these example files can be found in the repository in `examples/tutorials/simple_sine`_.

        **Exercise**

        Write a Calling Script with the following specifications:

        1. Set the generator function's lower and upper bounds to -6 and 6, respectively
        2. Increase the generator batch size to 10
        3. Set libEnsemble to stop execution after 160 *generations* using the ``gen_max`` option
        4. Print an error message if any errors occurred while libEnsemble was running

        .. dropdown:: **Click Here for Solution**

            .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial_2.py
                :language: python
                :linenos:
                :emphasize-lines: 15,16,17,27,33,34

    .. tab-item:: 5. Next steps

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

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial_3.py
            :language: python
            :start-at: # libE_specs = LibeSpecs
            :end-at: # libE_specs = LibeSpecs

        We'll be parameterizing our MPI runtime with a ``parse_args=True`` argument to
        the ``Ensemble`` class instead of ``libE_specs``. We'll also use an ``ensemble.is_manager``
        attribute so only the first MPI rank runs the data-processing code.

        The bottom of your calling script should now resemble:

        .. literalinclude:: ../../libensemble/tests/functionality_tests/test_local_sine_tutorial_3.py
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
        :doc:`platform guides <../platforms/platforms_index>` for introductions to using
        libEnsemble on many such machines.

        libEnsemble's Executors can launch non-Python user applications and simulations across
        allocated compute resources. Try out this feature with a more-complicated
        libEnsemble use-case within our
        :doc:`Electrostatic Forces tutorial <./executor_forces_tutorial>`.

.. _Matplotlib: https://matplotlib.org/
.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface
.. _MPICH: https://www.mpich.org/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/install.html
.. _NumPy: https://www.numpy.org/
.. _here: https://www.mpich.org/downloads/
.. _examples/tutorials/simple_sine: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/simple_sine
.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target:  http://colab.research.google.com/github/Libensemble/libensemble/blob/develop/examples/tutorials/simple_sine/sine_tutorial_notebook.ipynb
