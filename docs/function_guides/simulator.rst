.. _funcguides-sim:

Simulator Functions
===================

Simulator and :ref:`Generator functions<funcguides-gen>` have relatively similar interfaces.

.. code-block:: python

    def my_simulation(Input, persis_info, sim_specs, libE_info):
        batch_size = sim_specs["user"]["batch_size"]

        Output = np.zeros(batch_size, sim_specs["out"])
        ...
        Output["f"], persis_info = do_a_simulation(Input["x"], persis_info)

        return Output, persis_info

Most ``sim_f`` function definitions written by users resemble::

    def my_simulation(Input, persis_info, sim_specs, libE_info):

where:

    * ``Input`` is a selection of the :ref:`History array<funcguides-history>`
    * :ref:`persis_info<datastruct-persis-info>` is a dictionary containing state information
    * :ref:`sim_specs<datastruct-sim-specs>` is a dictionary of simulation parameters, including which fields from the History array got sent
    *  ``libE_info`` is a dictionary containing libEnsemble-specific entries

Valid simulator functions can accept a subset of the above parameters. So a very simple simulator function can start::

    def my_simulation(Input):

If sim_specs was initially defined::

    sim_specs = {
        "sim_f": some_function,
        "in": ["x"],
        "out:" ["f", float, (1,)],
        "user": {
            "batch_size": 128
        }
    }

Then user parameters and a *local* array of outputs may be obtained/initialized like::

    batch_size = sim_specs["user"]["batch_size"]
    Output = np.zeros(batch_size, dtype=sim_specs["out"])

This array should be populated with output values from the simulation::

    Output["f"], persis_info = do_a_simulation(Input["x"], persis_info)

Then return the array and ``persis_info`` to libEnsemble::

    return Output, persis_info

Between the ``Output`` definition and the ``return``, any level and complexity
of computation can be performed. Users are encouraged to use the :doc:`executor<../executor/overview>`
to submit applications to parallel resources if necessary, or plug in components from
other libraries to serve their needs.

Executor
--------

libEnsemble's Executors are commonly used within simulator functions to launch
and monitor applications. An excellent overview is already available
:doc:`here<../executor/overview>`.

See the :doc:`Executor with Electrostatic Forces tutorial<../tutorials/executor_forces_tutorial>`
for an additional example to try out.

Persistent Simulators
---------------------

Although comparatively uncommon, simulator functions can also be written
in a persistent fashion. See the :ref:`here<persistent-gens>` for a general API overview
of writing persistent generators, since the interface is largely identical. The only
differences are to pass ``EVAL_SIM_TAG`` when instantiating a ``PersistentSupport``
class instance and to return ``FINISHED_PERSISTENT_SIM_TAG`` when the simulator
function returns.

.. note::
  An example routine using a persistent simulator can be found in test_persistent_sim_uniform_sampling_.

.. _test_persistent_sim_uniform_sampling: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/regression_tests/test_persistent_sim_uniform_sampling.py
