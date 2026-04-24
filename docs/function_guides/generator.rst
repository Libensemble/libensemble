.. _funcguides-gen:

Generators
==========

Writing a Generator
-------------------

.. note::
    The `gest-api` generator interface is the recommended approach for new libEnsemble projects.
    The "Legacy Generator Function" interface is supported for backward compatibility but may be deprecated in a future release.

.. tab-set::

    .. tab-item:: Standardized Generator (gest-api)

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

    .. tab-item:: Legacy Generator Function

        .. code-block:: python

            def my_generator(Input, persis_info, gen_specs, libE_info):
                batch_size = gen_specs["user"]["batch_size"]

                Output = np.zeros(batch_size, gen_specs["out"])
                # ...
                Output["x"], persis_info = generate_next_simulation_inputs(Input["f"], persis_info)

                return Output, persis_info

        Most ``gen_f`` function definitions written by users resemble::

            def my_generator(Input, persis_info, gen_specs, libE_info):

        where:

            * ``Input`` is a selection of the :ref:`History array<funcguides-history>`, a NumPy structured array.
            * :ref:`persis_info<datastruct-persis-info>` is a dictionary containing state information.
            * :ref:`gen_specs<datastruct-gen-specs>` is a dictionary of generator parameters.
            *  ``libE_info`` is a dictionary containing miscellaneous entries.

        Valid generator functions can accept a subset of the above parameters. So a very simple generator can start::

            def my_generator(Input):

        If ``gen_specs`` was initially defined:

        .. code-block:: python

            gen_specs = GenSpecs(
                gen_f=my_generator,
                inputs=["f"],
                outputs=["x", float, (1,)],
                user={"batch_size": 128},
            )

        Then user parameters and a *local* array of outputs may be obtained/initialized like::

            batch_size = gen_specs["user"]["batch_size"]
            Output = np.zeros(batch_size, dtype=gen_specs["out"])

        This array should be populated by whatever values are generated within
        the function::

            Output["x"], persis_info = generate_next_simulation_inputs(Input["f"], persis_info)

        Then return the array and ``persis_info`` to libEnsemble::

            return Output, persis_info

        Between the ``Output`` definition and the ``return``, any computation can be performed.
        Users can try an :doc:`executor<../executor/overview>` to submit applications to parallel
        resources, or plug in components from other libraries to serve their needs.

        .. note::

            State ``gen_f`` information like checkpointing should be
            appended to ``persis_info``.

        .. _persistent-gens:

        Persistent Generators
        ---------------------

        While non-persistent generators return after completing their calculation, persistent
        generators do the following in a loop:

            1. Receive simulation results and metadata; exit if metadata instructs.
            2. Perform analysis.
            3. Send subsequent simulation parameters.

        Persistent generators don't need to be re-initialized on each call, but are typically
        more complicated. The persistent :doc:`APOSMM<../examples/aposmm>`
        optimization generator function included with libEnsemble maintains
        local optimization subprocesses based on results from complete simulations.

        Use ``GenSpecs.persis_in`` to specify fields to send back to the generator throughout the run.
        ``GenSpecs.inputs`` only describes the input fields when the function is **first called**.

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

        By default, a persistent worker is expected to
        receive and send data in a *ping pong* fashion. Alternatively,
        a worker can be initiated in *active receive* mode by the allocation
        function (see :ref:`start_only_persistent<start_only_persistent_label>`).
        The persistent worker can then send and receive from the manager at any time.

        Ensure there are no communication deadlocks in this mode. In manager-worker message exchanges, only the worker-side
        receive is blocking by default (a non-blocking option is available).

        Cancelling Simulations
        ----------------------

        Previously submitted simulations can be cancelled by sending a message to the manager:

        .. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
        .. autofunction:: request_cancel_sim_ids

        - If a generated point is cancelled by the generator **before sending** to another worker for simulation, then it won't be sent.
        - If that point has **already been evaluated** by a simulation, the ``cancel_requested`` field will remain ``True``.
        - If that point is **currently being evaluated**, a kill signal will be sent to the corresponding worker; it must be manually processed in the simulation function.

        The :doc:`Borehole Calibration tutorial<../tutorials/calib_cancel_tutorial>` gives an example
        of the capability to cancel pending simulations.

        Modification of existing points
        -------------------------------

        To change existing fields of the History array, create a NumPy structured array where the ``dtype`` contains
        the ``sim_id`` and the fields to be modified. Send this array with ``keep_state=True`` to the manager.
        This will overwrite the manager's History array.

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
        the ensemble as soon as a persistent generator returns. The usual return values should be given.

        Examples
        --------

        Examples of non-persistent and persistent generator functions
        can be found :doc:`here<../examples/gen_funcs>`.
