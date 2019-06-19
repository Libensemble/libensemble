==========================
Simple Local Sine Tutorial
==========================

This introductory tutorial demonstrates the capability to perform ensembles of
calculations in parallel using :doc:`libEnsemble<../quickstart>` with Python's Multiprocessing.

The foundation of writing libEnsemble routines is accounting for four components:

    1. The *Generator Function* :ref:`gen_f<api_gen_f>`, which produces values for simulations.
    2. The *Simulator Function* :ref:`sim_f<api_sim_f>`, which performs simulations based on values from ``gen_f``.
    3. The *Allocation Function* :ref:`alloc_f<api_alloc_f>`, which decides which of the previous two functions should be called, when.
    4. The *Calling Script*, which defines parameters and information about these functions and the libEnsemble task, then begins execution.

libEnsemble initializes a *manager* process and as many *worker*
processes as the user requests. These workers control and monitor jobs of widely
varying sizes and capabilities, including calling ``gen_f`` and ``sim_f`` functions, and
passing values between them.

For this tutorial, our ``gen_f`` will produce uniform randomly-sampled values,
and our ``sim_f`` will be tasked with finding the sine of each. Thankfully,
we don't need to specify a new allocation function by default. All generated and simulated
values alongside other parameters are stored in :ref:`H<datastruct-history-array>`,
the History array.


.. _libEnsemble: https://libensemble.readthedocs.io/en/latest/quickstart.html

Getting started
---------------

First, create an empty directory to store our code. Make sure Python 3 is
installed.

.. code-block:: bash

    $ mkdir libe_tutorial
    $ cd libe_tutorial
    $ python3 --version
    Python 3.6.0            # This should be >= 3.4

For this tutorial, you need NumPy_ to perform the calculations and (optionally)
Matplotlib_ to visualize your results. Install libEnsemble and these other libraries
with:

.. code-block:: bash

    $ pip3 install numpy
    $ pip3 install libensemble
    $ pip3 install matplotlib # Optional

.. _NumPy: https://www.numpy.org/
.. _Matplotlib: https://matplotlib.org/

Generator function
------------------

We'll start the coding portion of this tutorial by writing our :ref:`gen_f<api_gen_f>`, or generator
function.

An available libEnsemble worker will call this generator function with the following parameters:

* :ref:`H<datastruct-history-array>`: The History array. Updated by the workers with ``gen_f`` and ``sim_f`` inputs and outputs, then returned to the user. libEnsemble passes ``H`` to the generator function for users who may want to generate new values based on previous values.

* :ref:`persis_info<datastruct-persis-info>`: Dictionary with worker-specific information. In our case this dictionary contains random streams for generating random numbers.

* :ref:`gen_specs<datastruct-gen-specs>`: Dictionary with entries like simulation IDs, inputs and outputs, data-types, and other specifications for the generator function.

Later on we'll write ``gen_specs`` and ``persis_info`` explicitly in our calling script.

For now, create a new Python file named 'generator.py'. Write the following:

.. code-block:: python
    :linenos:

    import numpy as np


    def gen_random_sample(H, persis_info, gen_specs, _):
        # underscore parameter for internal/testing arguments

        # Get lower and upper bounds from gen_specs
        lower = gen_specs['lower']
        upper = gen_specs['upper']

        # Determine how many values to generate
        num = len(lower)
        batch_size = gen_specs['gen_batch_size']

        # Create array of 'batch_size' zeros
        out = np.zeros(batch_size, dtype=gen_specs['out'])

        # Replace those zeros with the random numbers
        out['x'] = persis_info['rand_stream'].uniform(lower, upper, (batch_size, num))

        # Send back our output and persis_info
        return out, persis_info


Our function creates 'batch_size' random numbers uniformly distributed
between the 'lower' and 'upper' bounds. A random stream
from ``persis_info`` is used to generate these values. Finally, the values are placed
into a NumPy array that meets the specifications from ``gen_specs['out']``.


Simulator function
------------------

Next, we'll write our :ref:`sim_f<api_sim_f>` or simulator function. Simulator
functions perform calculations based on the values output by the generator function.
The only new parameter here is :ref:`sim_specs<datastruct-sim-specs>`, which serves
a similar purpose to ``gen_specs``.

Create a new Python file named 'simulator.py'. Write the following:

.. code-block:: python
    :linenos:

    import numpy as np


    def sim_find_sine(H, persis_info, sim_specs, _):
        # underscore for internal/testing arguments

        # Create an output array of a single zero
        out = np.zeros(1, dtype=sim_specs['out'])

        # Set the zero to the sine of the input value stored in H
        out['y'] = np.sin(H['x'])

        # Send back our output and persis_info
        return out, persis_info

Our simulator function is called by a worker for every value in it's batch from the
generator function. This function calculates the sine of the passed value, then returns
it so a worker can log it into ``H``.


Calling Script
--------------

Now we can write the calling script that configures our generator and simulator
functions and calls libEnsemble.

Create an empty Python file named 'calling_script.py'.
In this file, we'll start by importing NumPy, libEnsemble, and the generator and
simulator functions we just created.

Next, in a dictionary called :ref:`libE_specs<datastruct-libe-specs>` we'll specify
the number of workers and the type of manager/worker communication libEnsemble will
use. Our communication method, referred to by 'comms', is 'local' because we're
using Python's multiprocessing.

.. code-block:: python
    :linenos:

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_random_sample
    from simulator import sim_find_sine

    nworkers = 4
    libE_specs = {'nprocesses': nworkers, 'comms': 'local'}

Our calling script is where we outline the settings and specifications
for our generator and simulator functions in the :ref:`gen_specs<datastruct-gen-specs>`
and :ref:`sim_specs<datastruct-sim-specs>` dictionaries that we saw previously.
These dictionaries also describe to libEnsemble what inputs and outputs from those
functions to expect.

.. code-block:: python
    :linenos:

    gen_specs = {'gen_f': gen_random_sample,      # Our generator function
                 'in': ['sim_id'],                  # Input field names. 'sim_id' necessary default
                 'out': [('x', float, (1,))],       # gen_f output (name, type, size).
                 'lower': np.array([-3]),           # lower boundary for random sampling.
                 'upper': np.array([3]),            # upper boundary for random sampling.
                 'gen_batch_size': 5}               # number of values gen_f will generate per call

    sim_specs = {'sim_f': sim_find_sine,          # Our simulator function
                 'in': ['x'],                       # Input field names. 'x' from gen_f output
                 'out': [('y', float)]}             # sim_f output. 'y' = sine('x')


Recall that each worker is assigned an entry in the :ref:`persis_info<datastruct-persis-info>` dictionary that, in our case, contains  a ``RandomState()``
stream for uniform random sampling. We populate that dictionary here. Finally, we specify
the circumstances where libEnsemble should stop execution in :ref:`exit_criteria<datastruct-exit-criteria>`.

.. code-block:: python
    :linenos:

    persis_info = {}

    for i in range(1, nworkers+1):                # Worker numbers start at 1.
        persis_info[i] = {
            'rand_stream': np.random.RandomState(i),
            'worker_num': i}

    exit_criteria = {'sim_max': 80}               # Stop libEnsemble after 80 simulations

Now we're ready to write our libEnsemble :doc:`libE<../libE_module>` function call.
This :ref:`H<datastruct-history-array>` is the final version of the History array. 'flag' should be zero if no
errors occur.

.. code-block:: python
    :linenos:

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                              libE_specs=libE_specs)

    print([i for i in H.dtype.fields])            # Some (optional) statements to visualize our History array
    print(H)


Now that all these files are completed, we can run our simulation.

.. code-block:: bash

  $ python3 calling_script.py

If everything ran perfectly, You should get something similar to the following output
for ``H``. The columns might be rearranged.

.. code-block::

  ['y', 'given_time', 'gen_worker', 'sim_worker', 'given', 'returned', 'x', 'allocated', 'sim_id', 'gen_time']
  [(-0.37466051, 1.55968252e+09, 2, 2,  True,  True, [-0.38403059],  True,  0, 1.55968252e+09)
  (-0.29279634, 1.55968252e+09, 2, 3,  True,  True, [-2.84444261],  True,  1, 1.55968252e+09)
  ( 0.29358492, 1.55968252e+09, 2, 4,  True,  True, [ 0.29797487],  True,  2, 1.55968252e+09)
  (-0.3783986 , 1.55968252e+09, 2, 1,  True,  True, [-0.38806564],  True,  3, 1.55968252e+09)
  (-0.45982062, 1.55968252e+09, 2, 2,  True,  True, [-0.47779319],  True,  4, 1.55968252e+09)
  ...

In this arrangement, our output values are listed on the far-left with the generated
values being the fourth column from the right. Again, your columns might be rearranged.

Two additional log files should also have been created.
'ensemble.log' contains debugging or informational logging output from libEnsemble,
while 'libE_stats.txt' contains a quick summary of all calculations performed.

I graphed my output using Matplotlib, coloring entries by which worker performed
the simulation:

.. image:: ../images/sinex.png
  :alt: sine

If you want to verify your results through plotting and you installed Matplotlib
earlier, copy and paste the following code into the bottom of your calling script
and run ``python3 calling_script.py`` again

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
  plt.show()




Next Steps
----------

Coming soon


FAQ
---

Coming soon
