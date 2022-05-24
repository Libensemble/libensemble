==================================
Multiprocessing with a Simple Sine
==================================

This introductory tutorial demonstrates the capability to perform ensembles of
calculations in parallel using :doc:`libEnsemble<../introduction>` with Python's
multiprocessing.

The foundation of writing libEnsemble routines is accounting for at least three components:

    1. A :ref:`generator function<api_gen_f>`, that produces values for simulations
    2. A :ref:`simulator function<api_sim_f>`, that performs simulations based on values from the generator function
    3. A :doc:`calling script<../libe_module>`, for defining settings, fields, and functions, then starting the run

libEnsemble initializes a *manager* process and as many *worker* processes as the
user requests. The manager (via the :ref:`allocation function<api_alloc_f>`)
coordinates data transfer between workers and assigns units of work to each worker,
consisting of a function to run and accompanying data. These functions perform their
work in-line with Python and/or launch and control user applications with
libEnsemble's :ref:`Executors<executor_index>`. Workers pass results back to the manager.

For this tutorial, we'll write our generator and simulator functions entirely in Python
without other applications. Our generator will produce uniform randomly sampled
values, and our simulator will calculate the sine of each. By default we don't
need to write a new allocation function. All generated and simulated values
alongside other parameters are stored in :ref:`H<datastruct-history-array>`,
the history array.

Getting started
---------------

libEnsemble and its functions are written entirely in Python_. Let's make sure
the correct version is installed.

.. code-block:: bash

    $ python --version
    Python 3.7.0            # This should be >= 3.7

.. _Python: https://www.python.org/

For this tutorial, you need NumPy_ to perform calculations and (optionally)
Matplotlib_ to visualize your results. Install libEnsemble and these other
libraries with

.. code-block:: bash

    $ pip install numpy
    $ pip install libensemble
    $ pip install matplotlib # Optional

If your system doesn't allow you to perform these installations, try adding
``--user`` to the end of each command.

.. _NumPy: https://www.numpy.org/
.. _Matplotlib: https://matplotlib.org/

Generator function
------------------

Let's begin the coding portion of this tutorial by writing our generator function,
or :ref:`gen_f<api_gen_f>`.

An available libEnsemble worker will call this generator function with the
following parameters:

* :ref:`H_in<datastruct-history-array>`: A selection of the History array, a NumPy structured array
  for storing information about each point generated and processed in the ensemble.
  libEnsemble passes a selection of ``H`` to the generator function in case the user
  wants to generate new values based on previous data.

* :ref:`persis_info<datastruct-persis-info>`: Dictionary with worker-specific
  information. In our case, this dictionary contains NumPy Random Stream objects
  for generating random numbers.

* :ref:`gen_specs<datastruct-gen-specs>`: Dictionary with user-defined fields and
  parameters for the generator. Customizable parameters such as boundaries and batch
  sizes are placed within the ``gen_specs['user']`` dictionary, while input/output fields
  and other specifications that libEnsemble depends on to operate the generator are
  placed outside ``user``.

Later on, we'll populate ``gen_specs`` and ``persis_info`` when we initialize libEnsemble.

For now, create a new Python file named ``generator.py``. Write the following:

.. code-block:: python
    :linenos:
    :caption: examples/tutorials/simple_sine/tutorial_gen.py

    import numpy as np

    def gen_random_sample(H_in, persis_info, gen_specs, _):
        # Underscore ignores advanced arguments

        # Pull out user parameters
        user_specs = gen_specs['user']

        # Get lower and upper bounds
        lower = user_specs['lower']
        upper = user_specs['upper']

        # Determine how many values to generate
        num = len(lower)
        batch_size = user_specs['gen_batch_size']

        # Create empty array of 'batch_size' zeros. Array dtype should match 'out' fields
        out = np.zeros(batch_size, dtype=gen_specs['out'])

        # Set the 'x' output field to contain random numbers, using random stream
        out['x'] = persis_info['rand_stream'].uniform(lower, upper, (batch_size, num))

        # Send back our output and persis_info
        return out, persis_info

Our function creates ``batch_size`` random numbers uniformly distributed
between the ``lower`` and ``upper`` bounds. A random stream
from ``persis_info`` is used to generate these values, which are then placed
into an output NumPy array that meets the specifications from ``gen_specs['out']``.

Exercise
^^^^^^^^

Write a simple generator function that instead produces random integers, using
the ``numpy.random.Generator.integers(low, high, size)`` function.

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

       import numpy as np

       def gen_random_ints(H_in, persis_info, gen_specs, _):

           user_specs = gen_specs['user']
           lower = user_specs['lower']
           upper = user_specs['upper']
           num = len(lower)
           batch_size = user_specs['gen_batch_size']

           out = np.zeros(batch_size, dtype=gen_specs['out'])
           out['x'] = persis_info['rand_stream'].integers(lower, upper, (batch_size, num))

           return out, persis_info

Simulator function
------------------

Next, we'll write our simulator function or :ref:`sim_f<api_sim_f>`. Simulator
functions perform calculations based on values from the generator function.
The only new parameter here is :ref:`sim_specs<datastruct-sim-specs>`, which
serves a purpose similar to the ``gen_specs`` dictionary.

Create a new Python file named ``simulator.py``. Write the following:

.. code-block:: python
    :linenos:
    :caption: examples/tutorials/simple_sine/tutorial_sim.py

    import numpy as np

    def sim_find_sine(H_in, persis_info, sim_specs, _):
        # underscore for internal/testing arguments

        # Create an output array of a single zero
        out = np.zeros(1, dtype=sim_specs['out'])

        # Set the zero to the sine of the input value stored in H
        out['y'] = np.sin(H_in['x'])

        # Send back our output and persis_info
        return out, persis_info

Our simulator function is called by a worker for every work item produced by
the generator function. This function calculates the sine of the passed value,
then returns it so a worker can log it into ``H``.

Exercise
^^^^^^^^

Write a simple simulator function that instead calculates the *cosine* of a received
value, using the ``numpy.cos(x)`` function.

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

       import numpy as np

       def sim_find_cosine(H_in, persis_info, gen_specs, _):

        out = np.zeros(1, dtype=sim_specs['out'])

        out['y'] = np.cos(H_in['x'])

        return out, persis_info

Calling Script
--------------

Now we can write the calling script that configures our generator and simulator
functions and calls libEnsemble.

Create an empty Python file named ``calling_script.py``.
In this file, we'll start by importing NumPy, libEnsemble, and the generator and
simulator functions we just created.

Next, in a dictionary called :ref:`libE_specs<datastruct-libe-specs>` we'll
specify the number of workers and the type of manager/worker communication
libEnsemble will use. Our communication method, ``'local'``, refers to Python's
multiprocessing.

.. code-block:: python
    :linenos:

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_random_sample
    from simulator import sim_find_sine
    from libensemble.tools import add_unique_random_streams

    nworkers = 4
    libE_specs = {'nworkers': nworkers, 'comms': 'local'}

We configure the settings and specifications for our ``sim_f`` and ``gen_f``
functions in the :ref:`gen_specs<datastruct-gen-specs>` and
:ref:`sim_specs<datastruct-sim-specs>` dictionaries, which we saw previously
being passed to our functions. These dictionaries also describe to libEnsemble
what inputs and outputs from those functions to expect.

.. code-block:: python
    :linenos:

    gen_specs = {'gen_f': gen_random_sample,   # Our generator function
                 'out': [('x', float, (1,))],  # gen_f output (name, type, size)
                 'user': {
                    'lower': np.array([-3]),   # lower boundary for random sampling
                    'upper': np.array([3]),    # upper boundary for random sampling
                    'gen_batch_size': 5        # number of x's gen_f generates per call
                    }
                 }

    sim_specs = {'sim_f': sim_find_sine,       # Our simulator function
                 'in': ['x'],                  # Input field names. 'x' from gen_f output
                 'out': [('y', float)]}        # sim_f output. 'y' = sine('x')

Recall that each worker is assigned an entry in the
:ref:`persis_info<datastruct-persis-info>`  dictionary that, in this tutorial,
contains  a ``RandomState()`` random stream for uniform random sampling. We
populate that dictionary here using a utility from the
:doc:`tools module<../utilities>`. We then specify the circumstances where
libEnsemble should stop execution in :ref:`exit_criteria<datastruct-exit-criteria>`.

.. code-block:: python
    :linenos:

    persis_info = add_unique_random_streams({}, nworkers+1) # Worker numbers start at 1

    exit_criteria = {'sim_max': 80}           # Stop libEnsemble after 80 simulations

Now we're ready to write our libEnsemble :doc:`libE<../programming_libE>`
function call. This :ref:`H<datastruct-history-array>` is the final version of
the history array. ``flag`` should be zero if no errors occur.

.. code-block:: python
    :linenos:

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                libE_specs=libE_specs)

    print([i for i in H.dtype.fields])  # (optional) to visualize our history array
    print(H)

That's it! Now that these files are complete, we can run our simulation.

.. code-block:: bash

  $ python calling_script.py

If everything ran perfectly and you included the above print statements, you
should get something similar to the following output for ``H`` (although the
columns might be rearranged).

.. code-block::

  ['y', 'sim_started_time', 'gen_worker', 'sim_worker', 'sim_started', 'sim_ended', 'x', 'allocated', 'sim_id', 'gen_ended_time']
  [(-0.37466051, 1.559+09, 2, 2,  True,  True, [-0.38403059],  True,  0, 1.559+09)
  (-0.29279634, 1.559+09, 2, 3,  True,  True, [-2.84444261],  True,  1, 1.559+09)
  ( 0.29358492, 1.559+09, 2, 4,  True,  True, [ 0.29797487],  True,  2, 1.559+09)
  (-0.3783986 , 1.559+09, 2, 1,  True,  True, [-0.38806564],  True,  3, 1.559+09)
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
script and run ``python calling_script.py`` again

.. code-block:: python
  :linenos:

  import matplotlib.pyplot as plt
  colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w']

  for i in range(1, nworkers + 1):
      worker_xy = np.extract(H['sim_worker'] == i, H)
      x = [entry.tolist()[0] for entry in worker_xy['x']]
      y = [entry for entry in worker_xy['y']]
      plt.scatter(x, y, label='Worker {}'.format(i), c=colors[i-1])

  plt.title('Sine calculations for a uniformly sampled random distribution')
  plt.xlabel('x')
  plt.ylabel('sine(x)')
  plt.legend(loc = 'lower right')
  plt.savefig('tutorial_sines.png')

Each of these example files can be found in the repository in `examples/tutorials/simple_sine`_.

Exercise
^^^^^^^^

Write a Calling Script with the following specifications:

  1. Use the :meth:`parse_args()<tools.parse_args>` function to detect ``nworkers`` and auto-populate ``libE_specs``
  2. Set the generator function's lower and upper bounds to -6 and 6, respectively
  3. Increase the generator batch size to 10
  4. Set libEnsemble to stop execution after 160 *generations* using the ``gen_max`` key
  5. Print an error message if any errors occurred while libEnsemble was running

.. container:: toggle

   .. container:: header

      **Click Here for Solution**

   .. code-block:: python
       :linenos:

       import numpy as np
       from libensemble.libE import libE
       from generator import gen_random_sample
       from simulator import sim_find_sine
       from libensemble.tools import add_unique_random_streams

       nworkers, is_manager, libE_specs, _ = parse_args()

       gen_specs = {'gen_f': gen_random_ints,
                    'out': [('x', float, (1,))],
                    'user': {
                       'lower': np.array([-6]),
                       'upper': np.array([6]),
                       'gen_batch_size': 10
                     }
                   }

       sim_specs = {'sim_f': sim_find_sine,
                     'in': ['x'],
                     'out': [('y', float)]}

       persis_info = add_unique_random_streams({}, nworkers+1)
       exit_criteria = {'gen_max': 160}

       H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                   libE_specs=libE_specs)

       if flag != 0:
          print('Oh no! An error occurred!')

Next steps
----------

libEnsemble with MPI
^^^^^^^^^^^^^^^^^^^^

MPI_ is a standard interface for parallel computing, implemented in libraries
such as MPICH_ and used at extreme scales. MPI potentially allows libEnsemble's
manager and workers to be distributed over multiple nodes and works in some
circumstances where Python's multiprocessing does not. In this section, we'll
explore modifying the above code to use MPI instead of multiprocessing.

We recommend the MPI distribution MPICH_ for this tutorial, which can be found
for a variety of systems here_. You also need mpi4py_, which can be installed
with ``pip install mpi4py``. If you'd like to use a specific version or
distribution of MPI instead of MPICH, configure mpi4py with that MPI at
installation with ``MPICC=<path/to/MPI_C_compiler> pip install mpi4py`` If this
doesn't work, try appending ``--user`` to the end of the command. See the
mpi4py_ docs for more information.

Verify that MPI has installed correctly with ``mpirun --version``.

Modifying the calling script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only a few changes are necessary to make our code MPI-compatible. Modify the top
of the calling script as follows:

.. code-block:: python
    :linenos:
    :emphasize-lines: 5,7,8,10,11

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_random_sample
    from simulator import sim_find_sine
    from libensemble.tools import add_unique_random_streams
    from mpi4py import MPI

    # nworkers = 4                                # nworkers will come from MPI
    libE_specs = {'comms': 'mpi'}                 # 'nworkers' removed, 'comms' now 'mpi'

    nworkers = MPI.COMM_WORLD.Get_size() - 1
    is_manager = (MPI.COMM_WORLD.Get_rank() == 0)  # manager process has MPI rank 0

So that only one process executes the graphing and printing portion of our code,
modify the bottom of the calling script like this:

.. code-block:: python
  :linenos:
  :emphasize-lines: 4

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                libE_specs=libE_specs)

    if is_manager:
        # Some (optional) statements to visualize our history array
        print([i for i in H.dtype.fields])
        print(H)

        import matplotlib.pyplot as plt
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w']

        for i in range(1, nworkers + 1):
            worker_xy = np.extract(H['sim_worker'] == i, H)
            x = [entry.tolist()[0] for entry in worker_xy['x']]
            y = [entry for entry in worker_xy['y']]
            plt.scatter(x, y, label='Worker {}'.format(i), c=colors[i-1])

        plt.title('Sine calculations for a uniformly sampled random distribution')
        plt.xlabel('x')
        plt.ylabel('sine(x)')
        plt.legend(loc='lower right')
        plt.savefig('tutorial_sines.png')

With these changes in place, our libEnsemble code can be run with MPI by

.. code-block:: bash

  $ mpirun -n 5 python calling_script.py

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

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface
.. _MPICH: https://www.mpich.org/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/install.html
.. _here: https://www.mpich.org/downloads/
.. _examples/tutorials/simple_sine: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials/simple_sine
