Standardized Simulator (gest-api)
=================================

**Introduction** \|\| **Standardized Simulator (gest-api)** \|\| `Legacy Simulator Function <simulator_legacy.html>`__

Standardized simulators are plain callables — no base class required — with the signature::

    def my_simulation(input_dict: dict, **kwargs) -> dict:

They receive a single point as a Python dictionary (keyed by VOCS variable and constant
names) and return a dictionary of outputs (keyed by VOCS objective, observable, and
constraint names).

.. code-block:: python

    def my_simulation(input_dict: dict, **kwargs) -> dict:
        x1 = input_dict["x1"]
        x2 = input_dict["x2"]
        f = (x1 - 1) ** 2 + (x2 - 2) ** 2
        return {"f": f}

Configure it with ``SimSpecs`` using a ``VOCS`` object. ``inputs`` and ``outputs``
are derived automatically from the VOCS when not set explicitly:

.. code-block:: python

    from gest_api.vocs import VOCS
    from libensemble.specs import SimSpecs

    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"f": "MINIMIZE"},
    )

    sim_specs = SimSpecs(
        simulator=my_simulation,
        vocs=vocs,
    )

If ``libE_info`` is needed (e.g., to access the :doc:`executor<../executor/ex_index>`),
declare it as a keyword argument and libEnsemble will pass it automatically::

    def my_simulation(input_dict: dict, libE_info=None, **kwargs) -> dict:
