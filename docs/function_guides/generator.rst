.. _funcguides-gen:

Generator Functions
===================

.. code-block:: python

    def my_generator(Input, persis_info, gen_specs):

        batch_size = gen_specs["user"]["batch_size"]

        Output = np.zeros(batch_size, gen_specs["out"])
        ...
        Output["x"], persis_info = generate_next_simulation_inputs(Input["f"], persis_info)

        return Output, persis_info

Most ``gen_f`` function definitions written by users resemble::

    def my_generator(Input, persis_info, gen_specs, libE_info):

Where:

    * ``Input`` is a selection of the :ref:`History array<funcguides-history>`
    * :ref:`persis_info<datastruct-persis-info>` is a dictionary containing state information
    * :ref:`gen_specs<datastruct-gen-specs>` is a dictionary of generator parameters, including which fields from the History array got sent
    *  ``libE_info`` is a dictionary containing libEnsemble-specific entries

Valid generator functions can accept a subset of the above parameters. So a very simple generator can start::

    def my_generator(Input):

If gen_specs was initially defined::

    gen_specs = {
        "gen_f": some_function,
        "in": ["f"],
        "out:" ["x", float, (1,)],
        "user": {
            "batch_size": 128
        }
    }

Then user parameters and a *local* array of outputs may be obtained/initialized like::

    batch_size = gen_specs["user"]["batch_size"]
    Output = np.zeros(batch_size, dtype=gen_specs["out"])

This array should be populated by whatever values are generated within
the function::

    Output["x"], persis_info = generate_next_simulation_inputs(Input["f"], persis_info)

Then return the array and ``persis_info`` to libEnsemble::

    return Output, persis_info

Between the ``Output`` definition and the ``return``, any level and complexity
of computation can be performed. Users are encouraged to use the :doc:`executor<../executor/overview>`
to submit applications to parallel resources if necessary, or plug in components from
other libraries to serve their needs.

.. note::

    State ``gen_f`` information like checkpointing should be
    appended to ``persis_info``.

.. _persistent-gens:

Persistent Generators
---------------------

While non-persistent generators return after completing their calculation, persistent
generators do the following in a loop:

    1. Receive simulation results and metadata. Exit if metadata instructs
    2. Perform analysis
    3. Send subsequent simulation parameters

Persistent generators don't need to be re-initialized on each call, but are typically
more complicated. The :doc:`APOSMM<../examples/aposmm>`
optimization generator function included with libEnsemble is persistent so it can
maintain multiple local optimization subprocesses based on results from complete simulations.

Use ``gen_specs["persis_in"]`` to specify fields to send back to the generator throughout runtime.
``gen_specs["in"]`` only describes the input fields when the function is **first called**.

Functions for a persistent generator to communicate directly with the manager
are available in the :ref:`libensemble.tools.persistent_support<p_gen_routines>` class.

Sending/receiving data is supported by the :ref:`PersistentSupport<p_gen_routines>` class::

    from libensemble.tools import PersistentSupport
    from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG

    my_support = PersistentSupport(libE_info, EVAL_GEN_TAG)

Implementing functions from the above class is relatively simple:

.. tab-set::

    .. tab-item:: send

        .. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
        .. autofunction:: send

        This function call typically resembles::

            my_support.send(local_H_out[selected_IDs])

        Note that this function has no return.

    .. tab-item:: recv

        .. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
        .. autofunction:: recv

        This function call typically resembles::

            tag, Work, calc_in = my_support.recv()

            if tag in [STOP_TAG, PERSIS_STOP]:
                cleanup()
                break

        The logic following the function call is typically used to break the persistent
        generator's main loop and return.

    .. tab-item:: send_recv

        .. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
        .. autofunction:: send_recv

        This function performs both of the previous functions in a single statement. Its
        usage typically resembles::

            tag, Work, calc_in = my_support.send_recv(local_H_out[selected_IDs])
            if tag in [STOP_TAG, PERSIS_STOP]:
                cleanup()
                break

        Once the persistent generator's loop has been broken because of
        the tag from the manager, it should return with an additional tag::

            return local_H_out, persis_info, FINISHED_PERSISTENT_GEN_TAG

See :ref:`calc_status<funcguides-calcstatus>` for more information about
the message tags.

.. _gen_active_recv:

Active receive mode
-------------------

By default, a persistent worker is expected to alternately
receive and send data in a *ping pong* fashion. Alternatively,
a worker can be initiated in *active receive* mode by the allocation
function (see :ref:`start_only_persistent<start_only_persistent_label>`).
The persistent worker can then send and receive from the manager at any time.

Ensure there are no communication deadlocks in this mode. In manager/worker message exchanges, only the worker-side
receive is blocking by default (a non-blocking option is available).

Cancelling Simulations
----------------------

Previously submitted simulations can be cancelled by sending a message to the manager.

To do this a PersistentSupport helper function is provided.

.. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
.. autofunction:: request_cancel_sim_ids

If a generated point is cancelled by the generator before it has been given to a
worker for evaluation, then it will never be given. If it has already returned from the
simulation, then results can be returned, but the ``cancel_requested`` field will remain ``True``.
However, if the simulation is already running, a kill signal will be sent to the worker.
This can be caught and acted upon by the simulation function, otherwise it will be ignored.

The :doc:`Borehole Calibration tutorial<../tutorials/calib_cancel_tutorial>` gives an example
of the capability to cancel pending simulations.

Modification of existing points
-------------------------------

To change existing fields of the History array, the generator can initialize an output
array where the *dtype* contains the ``sim_id`` and the fields to be modified (in
place of ``gen_specs["out"]``), and then send this output array to the manager.

This will overwrite the specific fields for the given *sim_ids*, as maintained by the manager.
If the changes do not correspond with newly generated points,
then the generator needs to tell the manager that it is not ready
to receive completed evaluations. Send to the manager with ``keep_state=True``.

For example, the cancellation function ``request_cancel_sim_ids`` could be replicated by
the following (where ``sim_ids_to_cancel`` is a list of integers):

.. code-block:: python

    # Send only these fields to existing H rows and libEnsemble will slot in the change.
    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[("sim_id", int), ("cancel_requested", bool)])
    H_o["sim_id"] = sim_ids_to_cancel
    H_o["cancel_requested"] = True
    ps.send(H_o, keep_state=True)

Generator initiated shutdown
----------------------------

If using a supporting allocation function, the generator can prompt the ensemble to shutdown
by simply exiting the function (e.g., on a test for a converged value). For example, the
allocation function :ref:`start_only_persistent<start_only_persistent_label>` closes down
the ensemble as soon a persistent generator returns. The usual return values should be given.

Examples
--------

Examples of non-persistent and persistent generator functions
can be found :doc:`here<../examples/gen_funcs>`.
