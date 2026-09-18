1. Getting started
==================

:doc:`Introduction <local_sine_tutorial>` || **1. Getting started** || :doc:`2. Generator <local_sine_tutorial_2>` || :doc:`3. Simulator <local_sine_tutorial_3>` || :doc:`4. Script <local_sine_tutorial_4>` || :doc:`5. Next steps <local_sine_tutorial_5>`

libEnsemble is written entirely in Python_. Let's make sure
the correct version is installed.

.. code-block:: bash

    python --version  # This should be >= 3.12

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
.. _NumPy: https://numpy.org/
