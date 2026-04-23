========================
Optimization with APOSMM
========================

This tutorial demonstrates libEnsemble's capability to identify multiple minima
of simulation output using the built-in :doc:`APOSMM<../examples/aposmm>`
(Asynchronously Parallel Optimization Solver for finding Multiple Minima)
:ref:`gen_f<api_gen_f>`. In this tutorial, we'll create a simple
simulation :ref:`sim_f<api_sim_f>` that defines a function with
multiple minima, then write a libEnsemble calling script that imports APOSMM and
parameterizes it to check for minima over a domain of outputs from our ``sim_f``.

|Open in Colab|

Six-Hump Camel Simulation Function
----------------------------------

Describing APOSMM's operations is simpler with a given function on which to
depict evaluations. We'll use the `Six-Hump Camel function`_, known to have six
global minima. A sample space of this function, containing all minima, appears
below:

    .. image:: ../images/basic_6hc.png
        :alt: Six-Hump Camel
        :scale: 60
        :align: center

Create a new Python file named ``six_hump_camel.py``. This will be our
simulator callable, incorporating the above function. Write the following:

.. code-block:: python
    :linenos:

    def six_hump_camel_func(x):
        """Six-Hump Camel function definition"""
        x1 = x["x1"]
        x2 = x["x2"]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2

        return {"f": term1 + term2 + term3}

APOSMM Operations
-----------------

APOSMM coordinates multiple local optimization runs starting from a collection
of sample points. These local optimization runs occur concurrently,
and can incorporate a variety of optimization methods, including from NLopt_,
`PETSc/TAO`_, SciPy_, or other external scripts.

Before APOSMM can start local optimization runs, some number of uniformly
sampled points must be evaluated (if no prior simulation evaluations are
provided). User-requested sample points can also be provided to APOSMM:

    .. image:: ../images/sampling_6hc.png
        :alt: Six-Hump Camel Sampling
        :scale: 60
        :align: center

Specifically, APOSMM will begin local optimization runs from evaluated points that
don't have points with smaller function values nearby (within a threshold
``r_k``). For the above example, after APOSMM receives the evaluations of the
uniformly sampled points, it will begin at most ``max_active_runs``  local
optimization runs.

As function values are returned to APOSMM, APOSMM gives them to the
corresponding local optimization runs so they can generate the next point(s) in
their runs; such points are then returned by APOSMM to the manager to be
evaluated by the simulation routine. As runs complete (i.e., a minimum is
found, or some termination criteria for the local optimization run is
satisfied), additional local optimization runs may be started or additional
uniformly sampled points may be evaluated. This continues until a ``STOP_TAG``
is sent by the manager, for example when the budget of simulation evaluations
has been exhausted, or when a sufficiently "good" simulation output has been
observed.

    .. image:: ../images/localopt_6hc.png
        :alt: Six-Hump Camel Local Optimization Points
        :scale: 60
        :align: center

Throughout, generated and evaluated points are appended to the
:ref:`History<funcguides-history>` array, with the field
``"local_pt"`` being ``True`` if the point is part of a local optimization run,
and ``"local_min"`` being ``True`` if the point has been ruled a local minimum.

Calling Script
--------------

Create a new Python file named ``my_first_aposmm.py``. Start by importing
libEnsemble classes, APOSMM, and our simulator callable:

.. code-block:: python
    :linenos:

    from six_hump_camel import six_hump_camel_func

    import libensemble.gen_funcs

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    from libensemble import Ensemble
    from libensemble.gen_classes import APOSMM
    from gest_api.vocs import VOCS
    from libensemble.specs import SimSpecs, GenSpecs, ExitCriteria

APOSMM supports a wide variety of external optimizers. The ``rc.aposmm_optimizers``
statement above indicates to APOSMM which optimization method package to use,
helping prevent unnecessary imports or package installations.

Next, initialize the ``Ensemble`` and define our variables and objectives using
a ``VOCS`` object:

.. code-block:: python
    :linenos:

    if __name__ == "__main__":
        workflow = Ensemble(parse_args=True)

        vocs = VOCS(
            variables={"x1": [-2, 2], "x2": [-1, 1], "x1_on_cube": [-2, 2], "x2_on_cube": [-1, 1]},
            objectives={"f": "MINIMIZE"},
        )

Notice the addition of ``x1_on_cube`` and ``x2_on_cube``. APOSMM requires variables scaled to the unit cube internally. By defining both sets of variables, APOSMM can translate between our actual domain and its internal domain.

Now, configure APOSMM. Because APOSMM internally uses variables named ``x``, ``x_on_cube``, and an objective named ``f``, we must map our ``VOCS`` fields to these internal names using ``variables_mapping``:

.. code-block:: python
    :linenos:

        aposmm = APOSMM(
            vocs,
            max_active_runs=workflow.nworkers,
            variables_mapping={"x": ["x1", "x2"], "x_on_cube": ["x1_on_cube", "x2_on_cube"], "f": ["f"]},
            initial_sample_size=100,
            localopt_method="scipy_Nelder-Mead",
            opt_return_codes=[0],
        )

        workflow.gen_specs = GenSpecs(
            generator=aposmm,
            vocs=vocs,
            batch_size=5,
            initial_batch_size=10,
        )

APOSMM is instantiated directly as a standardized generator. It handles its own required fields, simplifying our configurations. ``opt_return_codes`` is a list of integers that local optimization methods return when a minimum is detected. SciPy's Nelder-Mead returns 0.

Finally, we configure the simulation function, exit criteria, and run the workflow. We can also print out any points that APOSMM identified as local minima:

.. code-block:: python
    :linenos:

        workflow.sim_specs = SimSpecs(simulator=six_hump_camel_func, vocs=vocs)
        workflow.exit_criteria = ExitCriteria(sim_max=2000)

        H, _, _ = workflow.run()

        if workflow.is_manager:
            # We can map our variables back to an array for easy printing
            minima = [[row["x1"], row["x2"]] for row in H if row["local_min"]]
            print("Minima:", minima)

Final Setup, Run, and Output
----------------------------

If you haven't already, install SciPy so APOSMM can access the required
optimization method::

    pip install scipy

Finally, run this libEnsemble / APOSMM optimization routine with the following::

    python my_first_aposmm.py --nworkers 4

Please note that one worker will be "persistent" for APOSMM for the duration of
the routine.

After a couple seconds, the output should resemble the following::

    Minima: [[0.08988580227184285, -0.7126604246830723], [-0.08983226938927827, 0.7126622830878125], [-1.7036480556534283, 0.7960787201083437], [1.7035677028481488, -0.7961234727197022], [1.607106093246473, 0.5686524941018596], [-1.607102046898864, -0.568650772274404]]

The local minima for the Six-Hump Camel simulation function as evaluated by
APOSMM with libEnsemble should be listed directly above.

Please see the API reference :doc:`here<../examples/aposmm>` for
more APOSMM configuration options and other information.

Each of these example files can be found in the repository in `examples/tutorials/aposmm`_.

Applications
------------

APOSMM is not limited to evaluating minima from pure Python simulation functions.
Many common libEnsemble use-cases involve using
libEnsemble's :doc:`MPI Executor<../executor/overview>` to launch user
applications with parameters requested by APOSMM, then evaluate their output using
APOSMM, and repeat until minima are identified. A currently supported example
can be found in libEnsemble's `WarpX Scaling Test`_.

.. _examples/tutorials/aposmm: https://github.com/Libensemble/libensemble/tree/develop/examples/tutorials
.. _NLopt: https://nlopt.readthedocs.io/en/latest/
.. _PETSc/TAO: https://www.mcs.anl.gov/petsc/
.. _SciPy: https://scipy.org/
.. _Six-Hump Camel function: https://www.sfu.ca/~ssurjano/camel6.html
.. _WarpX Scaling Test: https://github.com/Libensemble/libe-community-examples/tree/main/warpx
.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target:  http://colab.research.google.com/github/Libensemble/libensemble/blob/develop/examples/tutorials/aposmm/aposmm_tutorial_notebook.ipynb
