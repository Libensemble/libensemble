==========================
Simple Local Sine Tutorial
==========================

This introductory tutorial demonstrates the capability to perform ensembles of
calculations in parallel using *libEnsemble* with Python's Multiprocessing.

The foundation of writing libEnsemble experiments is accounting for four major
components:

    * The *Generator Function* ``gen_f``, which produces values for simulations.
    * The *Simulator Function* ``sim_f``, which performs simulations based on values from ``gen_f``.
    * The *Allocation Function* ``alloc_f``, which decides which of the previous two functions should be called, when.
    * The *Calling Script*, which defines parameters and information about these functions and the libEnsemble task, then begins execution.

Each libEnsemble instance initializes a *manager* process and as many *worker*
processes as the user requests. These workers control and monitor jobs of widely
varying sizes and capabilities, including calling gen_f and sim_f functions, and
passing values between them.

For this tutorial, our ``gen_f`` will produce uniform randomly-sampled values,
and our ``sim_f`` will be simply tasked with finding the sine of each. We will
allow libEnsemble to use the default allocation function. By default, all generated
and simulated values alongside other parameters are stored in ``H``, the History
array.


Getting started
---------------

First, create an empty directory to store our code. Also make sure Python 3 is
installed.

.. code-block:: bash

    $ mkdir libe_tutorial
    $ cd libe_tutorial
    $ python3 --version
    Python 3.6.0            # This should be > 3.4

For this tutorial, you need NumPy to perform the calculations and (optionally)
Matplotlib to visualize your results. Install these with:

.. code-block:: bash

    $ pip install numpy
    $ pip install matplotlib  # Optional


Calling Script
--------------

Create a new Python file. Start by importing NumPy, libEnsemble, and our generator
and simulator functions ahead of time. Let's specify the number of workers and the
type of manager/worker communication libEnsemble will use. In our case, it's 'local'
because we're using Python's multiprocessing.

.. code-block:: python
    :linenos:

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_uniform_random_sample
    from simulator import sim_find_sine

    nworkers = 4
    libE_specs = {'nprocesses': nworkers, 'comms': 'local'}

This information in libE_specs is later passed to libEnsemble on execution.

Our calling script is the right spot to outline settings and specifications
for our generator and simulator functions that libEnsemble needs to run, stored
in ``gen_specs`` and ``sim_specs`` respectively. These dictionaries are used to
describe to libEnsemble what inputs and outputs from those functions to expect.
They are also passed to libEnsemble alongside ``libE_specs``.

.. code-block:: python
    :linenos:

    gen_specs = {'gen_f': gen_uniform_random_sample,  # Our generator function
               'in': ['sim_id'],                    # Input field names for 'gen_f'. 'sim_id' necessary default
               'out': [('x', float, (1,))],         # Gen output (name, type, size) saved in H. Sent by worker to sim_f
               'lower': np.array([-3]),             # (Optional) lower boundary for random sampling.
               'upper': np.array([3]),              # (Optional) upper boundary for random sampling.
               'gen_batch_size': 5}                 # (Optional) number of values gen_f will generate and pass to worker

    sim_specs = {'sim_f': sim_find_sine,              # Our simulator function
               'in': ['x'],                         # Input field names for sim_f. 'x' from generator output
               'out': [('y', float)]}               # 'y' = sine('x') . Simulator output saved in H

See the docs for more exact gen_specs and sim_specs information.

Each worker is assigned a ``persis_info`` dictionary that contains additional
persistent state information. In our case, each worker receives a ``RandomState()``
stream for uniform random sampling, which should hopefully prevent different workers
from receiving identical values from separate generator calls. Finally, we specify
the circumstances where libEnsemble should stop execution in ``exit_criteria``.

.. code-block:: python
    :linenos:

    persis_info = {}                                  # Dictionary of dictionaries

    for i in range(nworkers+1):                       # Worker numbers start at 1.
      persis_info[i] = {
          'rand_stream': np.random.RandomState(i),
          'worker_num': i}

    exit_criteria = {'sim_max': 80}                   # Stop libEnsemble after 80 simulations

Now we're ready to write our libEnsemble function call. ``H`` refers to the History
array populated throughout execution and returned at the end. It includes information
like which workers accessed gen_f and sim_f at what times, and with what data.
'flag' should be zero if no errors occur.

.. code-block:: python
    :linenos:

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                              libE_specs=libE_specs)

    print([i for i in H.dtype.fields])  # Some (optional) statements to visualize our History array
    print(H)

Before we run the above code, lets finish our generation and simulation functions.

Gen function
------------

An available worker will call our generator function, which creates ``batch``
random numbers uniformly distributed between the ``lower`` and ``upper`` bounds
from ``gen_specs``. The random state from ``persis_info`` is used to generate
these numbers, which are placed into a NumPy array with field-names and datatypes
that match those specified in ``gen_specs``.

Create a new Python file named ``generator.py``. Write the following:

.. code-block:: python
    :linenos:

    import numpy as np

    def gen_uniform_random_sample(H, persis_info, gen_specs, _):  # underscore for internal/testing arguments

        lower = gen_specs['lower']
        upper = gen_specs['upper']

        num = len(lower)                                # Should be 1, due to one-dimensional array being passed
        batch = gen_specs['gen_batch_size']             # How many values to generate each call by a worker

        out = np.zeros(batch, dtype=gen_specs['out'])   # Output array of 'batch' slots, with gen_specs specified data type
        out['x'] = persis_info['rand_stream'].uniform(lower, upper, (batch, num))

        return out, persis_info

Notice that H is included as a function argument. A user may want to build on previous
simulated or generated values (stored in H) to generate new values.

Sim function
------------

Our simulator function is called by a worker for every value in a batch from gen_f.
This function finds the sine of the passed value, then returns it so a worker
can log it into H.

Create a new Python file named ``simulator.py``. Write the following:

.. code-block:: python
    :linenos:

    import numpy as np

    def sim_find_sine(H, persis_info, sim_specs, _):

        out = np.zeros(1, dtype=sim_specs['out'])   # Similar output array
        out['y'] = np.sin(H['x'])
        return out, persis_info

Now that all these files are completed, we can run our simulation.

.. code-block:: bash

  $ python3 [calling script name].py

If everything ran perfectly, no errors should be output and libEnsemble shouldn't
produce any .pickle or .npy files (which contain a dump of H in the event of an
error). You should get something similar to the following output for H. The columns
might be rearranged.

.. code-block::

  ['y', 'given_time', 'gen_worker', 'sim_worker', 'given', 'returned', 'x', 'allocated', 'sim_id', 'gen_time']
  [(-0.37466051, 1.55968252e+09, 2, 2,  True,  True, [-0.38403059],  True,  0, 1.55968252e+09)
  (-0.29279634, 1.55968252e+09, 2, 3,  True,  True, [-2.84444261],  True,  1, 1.55968252e+09)
  ( 0.29358492, 1.55968252e+09, 2, 4,  True,  True, [ 0.29797487],  True,  2, 1.55968252e+09)
  (-0.3783986 , 1.55968252e+09, 2, 1,  True,  True, [-0.38806564],  True,  3, 1.55968252e+09)
  (-0.45982062, 1.55968252e+09, 2, 2,  True,  True, [-0.47779319],  True,  4, 1.55968252e+09)
  ...

In this arrangement, our output values are on the far-left with the generated values
being the fourth-column from the right. Again, your columns might be rearranged.

Two additional log files should also have been created.
``ensemble.log`` contains logging output from libEnsemble, while ``libE_stats.txt``
contains a quick summary of all calculations performed.

I graphed my output using Matplotlib, coloring entries by which worker performed
the simulation:

.. image:: ../images/sinex.png
  :alt: sine

If you want to try this plotting yourself, install Matplotlib, and paste the
following code into another python file:

.. code-block:: python
  :linenos:

  def plot(H, nworkers):
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

In your calling script, include this function then call it beneath the libEnsemble call:

.. code-block:: python

    plot(H, nworkers)

FAQ
---

Coming soon
