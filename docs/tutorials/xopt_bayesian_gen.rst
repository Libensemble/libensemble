Bayesian Optimization with Xopt generator
=========================================

**Requires**: libensemble, xopt, gest-api

This tutorial demonstrates using Xopt's Bayesian **ExpectedImprovementGenerator** with libEnsemble.
We'll show two approaches:
1. Using an xopt-style simulator (callable function)
2. Using a libEnsemble-style simulator function

Imports
-------

.. code-block:: python

    import numpy as np
    from gest_api.vocs import VOCS
    from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

    from libensemble import Ensemble
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
    from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

Simulator Function
------------------

First, we define the xopt-style simulator function.

This is a basic function just to show how it works.

.. code-block:: python

    def test_callable(input_dict: dict) -> dict:
        """Single-objective callable test function"""
        assert isinstance(input_dict, dict)
        x1 = input_dict["x1"]
        x2 = input_dict["x2"]
        y1 = x2
        c1 = x1
        return {"y1": y1, "c1": c1}

Setup
-----

Define the VOCS specification and set up the generator.

.. code-block:: python

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=4)

    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
        constraints={"c1": ["GREATER_THAN", 0.5]},
        constants={"constant1": 1.0},
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # Create 4 initial points and ingest them
    initial_points = [
        {"x1": 0.2, "x2": 2.0, "y1": 2.0, "c1": 0.2},
        {"x1": 0.5, "x2": 5.0, "y1": 5.0, "c1": 0.5},
        {"x1": 0.7, "x2": 7.0, "y1": 7.0, "c1": 0.7},
        {"x1": 0.9, "x2": 9.0, "y1": 9.0, "c1": 0.9},
    ]
    gen.ingest(initial_points)

Define libEnsemble specifications. Note the gen_specs and sim_specs are set using vocs.

Approach 1: Using Xopt-style Simulator (Callable Function)
-----------------------------------------------------------

The simulator is a simple callable function that takes a dictionary of inputs and returns a dictionary of outputs.

.. code-block:: python

    gen_specs = GenSpecs(
        generator=gen,
        vocs=vocs,
    )

    # Note: using 'simulator' parameter for xopt-style callable
    sim_specs = SimSpecs(
        simulator=test_callable,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=12)

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, _ = workflow.run()

    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        print(H[["x1", "x2", "y1", "c1"]])
        assert np.array_equal(H["y1"], H["x2"])
        assert np.array_equal(H["c1"], H["x1"])

Approach 2: Using libEnsemble-style Simulator Function
-------------------------------------------------------

Now we define the libEnsemble-style simulator function and use it in the workflow.

.. code-block:: python

    def test_sim(H, persis_info, sim_specs, _):
        """
        Simple sim function that takes x1, x2, constant1 from H and returns y1, c1.
        Logic: y1 = x2, c1 = x1
        """
        batch = len(H)
        H_o = np.zeros(batch, dtype=sim_specs["out"])

        for i in range(batch):
            x1 = H["x1"][i]
            x2 = H["x2"][i]
            H_o["y1"][i] = x2
            H_o["c1"][i] = x1

        return H_o, persis_info

Reset generator and change to libEnsemble-style simulator:

.. code-block:: python

    # Reset generator and change to libEnsemble-style simulator
    gen = ExpectedImprovementGenerator(vocs=vocs)
    gen.ingest(initial_points)

    gen_specs = GenSpecs(
        generator=gen,
        vocs=vocs,
    )

    # Note: using 'sim_f' parameter for libEnsemble-style function
    sim_specs = SimSpecs(
        sim_f=test_sim,
        vocs=vocs,
    )

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, _ = workflow.run()

    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        print(H[["x1", "x2", "y1", "c1"]])
        assert np.array_equal(H["y1"], H["x2"])
        assert np.array_equal(H["c1"], H["x1"])
