Standardized Generator (gest-api)
=================================

`Introduction <generator.html>`__ \|\| **Standardized Generator (gest-api)** \|\| `Legacy Generator Function <generator_legacy.html>`__

Standardized generators are classes that inherit from ``gest_api.Generator``.
They adhere to the ``gest-api`` standard and are parameterized by a ``VOCS``
object defining the problem's variables and objectives.

A basic generator implements the ``suggest()`` and ``ingest()`` methods, which
operate on lists of dictionaries:

.. code-block:: python
    :linenos:

    import numpy as np
    from gest_api import Generator
    from gest_api.vocs import VOCS


    class UniformSample(Generator):
        """Samples over the domain specified in the VOCS."""

        def __init__(self, vocs: VOCS):
            self.vocs = vocs
            self.rng = np.random.default_rng(1)
            super().__init__(vocs)

        def _validate_vocs(self, vocs):
            assert len(self.vocs.variable_names), "VOCS must contain variables."

        def suggest(self, n_trials):
            output = []
            for _ in range(n_trials):
                trial = {}
                for key in self.vocs.variables:
                    trial[key] = self.rng.uniform(self.vocs.variables[key].domain[0], self.vocs.variables[key].domain[1])
                output.append(trial)
            return output

        def ingest(self, calc_in):
            pass  # random sample so nothing to ingest

libEnsemble's handling of standardized generators is specified using ``GenSpecs``:

.. code-block:: python

    gen_specs = GenSpecs(
        generator=UniformSample(vocs),
        inputs=["sim_id"],
        persis_in=["x", "f"],
        outputs=[("x", float, 2)],
        vocs=vocs,
        user={"batch_size": 128},
    )

.. note::
    Ensure that ``gen_specs.inputs`` or ``gen_specs.persis_in`` requests at least one field
    (like ``"sim_id"`` or ``"f"``) to be sent back, even if the generator does not
    process them.
