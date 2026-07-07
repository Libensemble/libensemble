2. Generator
============

`Introduction <local_sine_tutorial.html>`__ \|\| `1. Getting started <local_sine_tutorial_1.html>`__ \|\| **2. Generator** \|\| `3. Simulator <local_sine_tutorial_3.html>`__ \|\| `4. Script <local_sine_tutorial_4.html>`__ \|\| `5. Next steps <local_sine_tutorial_5.html>`__

Let's begin the coding portion of this tutorial by writing our generator.

An available libEnsemble worker will call this generator's ``.suggest()`` method to obtain
new values to evaluate.

For now, create a new Python file named ``sine_gen.py``. Write the following:

.. literalinclude:: ../../../libensemble/tests/functionality_tests/sine_gen_std.py
    :language: python
    :linenos:
    :caption: examples/tutorials/simple_sine/sine_gen_std.py

libEnsemble accepts generators that implement the gest-api_ interface. These generators
accept a ``gest_api.VOCS`` object for configuration, and contain a ``.suggest(num_points)``
method that returns ``num_points`` points. Points consist of a list of dictionaries
with keys that match the variable names from the ``gest_api.VOCS`` object.

Our generator's ``suggest()`` method creates ``num_points`` dictionaries. For each key in
the generator's ``self.variables``, it creates a random number uniformly distributed
between the corresponding ``lower`` and ``upper`` bounds of its domain.

Our generator must implement a ``_validate_vocs()`` method. Here, we implement a simple
check that ensures the ``VOCS`` object has at least one variable.

.. _gest-api: https://github.com/campa-consortium/gest-api
