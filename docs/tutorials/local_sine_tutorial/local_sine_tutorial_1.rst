1. Getting started
==================

`Introduction <local_sine_tutorial.html>`__ \|\| **1. Getting started** \|\| `2. Generator <local_sine_tutorial_2.html>`__ \|\| `3. Simulator <local_sine_tutorial_3.html>`__ \|\| `4. Script <local_sine_tutorial_4.html>`__ \|\| `5. Next steps <local_sine_tutorial_5.html>`__

libEnsemble is written entirely in Python_. Let's make sure
the correct version is installed.

.. code-block:: bash

    python --version  # This should be >= 3.11

.. _Python: https://www.python.org/

For this tutorial, you need NumPy_ and (optionally)
Matplotlib_ to visualize your results. Install libEnsemble and these other
libraries with

.. code-block:: bash

    pip install libensemble
    pip install matplotlib # Optional

If your system doesn't allow you to perform these installations, try adding
``--user`` to the end of each command.

.. _Matplotlib: https://matplotlib.org/
.. _NumPy: https://www.numpy.org/
