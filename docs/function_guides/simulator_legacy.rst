Legacy Simulator Function
=========================

`Introduction <simulator.html>`__ \|\| `Standardized Simulator (gest-api) <simulator_standardized.html>`__ \|\| **Legacy Simulator Function**

.. code-block:: python

    def my_simulation(Input, persis_info, sim_specs, libE_info):
        batch_size = sim_specs["user"]["batch_size"]

        Output = np.zeros(batch_size, sim_specs["out"])
        # ...
        Output["f"], persis_info = do_a_simulation(Input["x"], persis_info)

        return Output, persis_info

Most ``sim_f`` function definitions written by users resemble::

    def my_simulation(Input, persis_info, sim_specs, libE_info):

where:

    * ``Input`` is a selection of the :ref:`History array<funcguides-history>`, a NumPy structured array.
    * :ref:`persis_info<datastruct-persis-info>` is a dictionary containing state information.
    * :ref:`sim_specs<datastruct-sim-specs>` is a dictionary of simulation parameters.
    *  ``libE_info`` is a dictionary containing libEnsemble-specific entries.

Valid simulator functions can accept a subset of the above parameters. So a very simple simulator function can start::

    def my_simulation(Input):

If ``sim_specs`` was initially defined:

.. code-block:: python

    sim_specs = SimSpecs(
        sim_f=my_simulation,
        inputs=["x"],
        outputs=["f", float, (1,)],
        user={"batch_size": 128},
    )

Then user parameters and a *local* array of outputs may be obtained/initialized like::

    batch_size = sim_specs["user"]["batch_size"]
    Output = np.zeros(batch_size, dtype=sim_specs["out"])

This array should be populated with output values from the simulation::

    Output["f"], persis_info = do_a_simulation(Input["x"], persis_info)

Then return the array and ``persis_info`` to libEnsemble::

    return Output, persis_info

Between the ``Output`` definition and the ``return``, any computation can be performed.
Users can try an :doc:`executor<../executor/ex_index>` to submit applications to parallel
resources, or plug in components from other libraries to serve their needs.
