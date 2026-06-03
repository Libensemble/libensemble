========================================
Calibration with Simulation Cancellation
========================================

Introduction - Calibration with libEnsemble and a Regression Model
------------------------------------------------------------------

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on instructions from a calibration Generator Function.
This capability is desirable, especially when evaluations are expensive, since
compute resources may then be more effectively applied toward critical evaluations.

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings, functions, and data fields within the calling script,
:ref:`persistent generator<persistent-gens>`, manager, and :ref:`sim_f<funcguides-sim>`
that make this capability possible, rather than outlining a step-by-step process.

The libEnsemble regression test ``test_persistent_surmise_calib.py`` demonstrates
cancellation of pending simulations, while the ``test_persistent_surmise_killsims.py``
test demonstrates libEnsemble's capability to also kill running simulations that
have been marked as cancelled.

Overview of the Calibration Problem
-----------------------------------

The generator class featured in this tutorial can be found in
``gen_classes/surmise_calib.py`` as ``SurmiseCalibrator``, a standardized ``gest-api``
generator that uses the `surmise`_ library for its calibration surrogate model interface.
The surmise library uses the "PCGPwM" emulation method in this example.

Say there is a computer model :math:`f(\theta, x)` to be calibrated.  To calibrate
is to find some parameter :math:`\theta_0` such that :math:`f(\theta_0, x)` closely
resembles data collected from a physical experiment.  For example, a (simple)
physical experiment may involve dropping a ball at different heights to study the
gravitational constant, and the corresponding computer model could be the set of
differential equations that govern the drop. In a case where the computation of
the computer model is relatively expensive, we employ a fast surrogate model to
approximate the model and to inform good parameters to test next.  Here the computer
model :math:`f(\theta, x)` is accessible only through performing :ref:`sim_f<funcguides-sim>`
evaluations.

As a convenience for testing, the ``observed`` data values are modelled by calling the ``sim_f``
for the known true theta, which in this case is the center of a unit hypercube. These values
are therefore stored at the start of libEnsemble's
main :doc:`History array<../history_output_logging>` array, and have associated ``sim_id``'s.

The generator then samples an initial batch of parameters
:math:`(\theta_1, \ldots, \theta_n)` and constructs a surrogate model.

For illustration, the initial batch of evaluations are arranged in the following sense:

.. math::

    \newcommand{\T}{\mathsf{T}}
    \mathbf{f} = \begin{pmatrix} f(\theta_1)^\T \\ \vdots \\ f(\theta_n)^\T \end{pmatrix}
    = \begin{pmatrix} f(\theta_1, x_1) & \ldots & f(\theta_1, x_m) \\ \vdots & \ddots & \vdots
    \\ f(\theta_n, x_1) & \ldots & f(\theta_n, x_m) \end{pmatrix}.

The surrogate then generates (suggests) new parameters for ``sim_f`` evaluations,
so the number of parameters :math:`n` grows as more evaluations are scheduled and performed.
As more evaluations are performed and ingested by the generator, the surrogate evolves and
suggests parameters closer to :math:`\theta_0` with uncertainty estimates.
The calibration can be terminated when exit criteria are met or the generator
determines it has found :math:`\theta_0` with some tolerance in the surrounding
uncertainty. At termination, the generator's ``finalize()`` method is called,
initiating the shutdown of the libEnsemble routine.

The ``SurmiseCalibrator`` class implements the standard ``suggest``/``ingest``
interface. The generator progresses through three phases:

1. **Observation phase**: ``suggest_numpy()`` returns points for the true theta to
   generate observation data. ``ingest_numpy()`` stores the results as observations.
2. **Initial batch phase**: ``suggest_numpy()`` returns a batch of initial theta
   samples. ``ingest_numpy()`` builds the emulator and calibrator from the results.
3. **Main loop phase**: ``ingest_numpy()`` updates tracking arrays and
   conditionally rebuilds the model. ``suggest_numpy()`` generates new thetas
   when sufficient results have arrived, and queues cancellation requests for
   obviated simulations via ``suggest_updates()``.

The following is pseudocode for the generator's ``suggest``/``ingest`` loop::

    phase 1 - suggest: generate observation points (true theta)
              ingest:  store observation results, build obs/obsvar
    phase 2 - suggest: generate initial theta batch
              ingest:  build emulator and calibrator
    phase 3 - repeat:
              ingest:  update tracking arrays from results
                       if rebuild condition met: update emulator, recalibrate
              suggest: if select condition met:
                           calib: generate new thetas from model
                           if any pending points must be obviated:
                               queue cancel requests via suggest_updates()
                           return new points
                       else: return empty (wait for more results)

Point Cancellation Requests and Dedicated Fields
------------------------------------------------

While the generator's ``suggest_numpy()`` generates new thetas from the model, it
simultaneously evaluates if any *pending* simulations ought to be cancelled
("obviated"). If so, the generator calls its internal ``_cancel_columns()``
method, which constructs cancellation arrays and queues them for the runner:

.. code-block:: python

    # Inside SurmiseCalibrator._suggest_main_loop():
    if _select_condition(self.pending):
        new_theta, info = select_next_theta(...)
        ...
        c_obviate = info["obviatesugg"]  # suggested columns to cancel
        if len(c_obviate) > 0:
            self._cancel_columns(c_obviate)

``_cancel_columns()`` iterates over the columns to cancel, and for each pending
point, appends its ``sim_id`` to a cancellation array. These arrays are returned
by ``suggest_updates()`` and sent to the manager with ``keep_state=True``:

.. code-block:: python

    def _cancel_columns(self, c_obviate):
        """Mark columns for cancellation and queue cancellation updates."""
        sim_ids_to_cancel = []
        columns = np.unique(c_obviate)
        for c in columns:
            col_offset = c * self.n_x
            for i in range(self.n_x):
                sim_id_cancel = self._obs_offset + col_offset + i
                if self.pending[i, c]:
                    sim_ids_to_cancel.append(sim_id_cancel)
                    self.pending[i, c] = 0

        if sim_ids_to_cancel:
            cancel_array = np.zeros(len(sim_ids_to_cancel), dtype=[("sim_id", int), ("cancel_requested", bool)])
            cancel_array["sim_id"] = sim_ids_to_cancel
            cancel_array["cancel_requested"] = True
            self._pending_cancellations.append(cancel_array)


    def suggest_updates(self):
        """Return pending cancellation updates."""
        updates = self._pending_cancellations
        self._pending_cancellations = []
        return updates

The ``LibensembleGenRunner`` sends these updates to the manager with
``keep_state=True``, which updates existing History rows (setting
``cancel_requested=True``) without changing the generator's active state.

In future calls to the allocation function by the manager, points that would have
been distributed for simulation work but are now marked with ``cancel_requested`` will not
be processed. The manager will send kill signals to workers that are already processing
cancelled points. These signals can be caught and acted on by the user ``sim_f``; otherwise
they will be ignored.

Allocation Function and Cancellation Configuration
--------------------------------------------------

The default allocation function ``only_persistent_gens`` is used automatically
with standardized generators. The relevant settings are passed via ``GenSpecs``:

.. code-block:: python

    gen_specs = GenSpecs(
        generator=generator,
        persis_in=["f", "sim_id"],
        out=gen_out,
        initial_batch_size=init_sample_size,
        async_return=True,
        active_recv_gen=True,
    )

For the kill-sims test, ``kill_canceled_sims`` is also enabled:

.. code-block:: python

    libE_specs["kill_canceled_sims"] = True

**async_return** tells the allocation function to return results to the generator as soon
as they come back from evaluation (once the initial sample is complete).

**initial_batch_size** gives the size of the initial sample that is batch returned to the gen.
This is calculated from other parameters in the calling script.

**active_recv_gen** allows the persistent generator to handle irregular communications (see below).

By default, workers (including persistent workers), are only
allocated work when they're in an :ref:`idle or non-active state<funcguides-workerarray>`.
However, since this generator must asynchronously update its model, the worker
running this generator remains in an *active receive* state, until it becomes
non-persistent. This means both the manager and persistent worker (generator in
this case) must be prepared for irregular sending/receiving of data.

Calling Script - Reading Results
--------------------------------

Within the libEnsemble calling script, once the ``Ensemble.run()`` method
has returned, it's a simple enough process to view the History rows
that were marked as cancelled::

    H, _, _ = test.run()

    if test.is_manager:
        print("Cancelled sims", H["sim_id"][H["cancel_requested"]])

Here's an example graph showing the relationship between scheduled, cancelled (obviated),
failed, and completed simulations requested by the ``gen_f``. Notice that for each
batch of scheduled simulations, most either complete or fail but the rest are
successfully obviated:

    .. image:: ../images/gen_v_fail_or_cancel.png
      :width: 600
      :alt: surmise_sample_graph
      :align: center

Please see the ``test_persistent_surmise_calib.py`` regression test for an example
routine using the ``SurmiseCalibrator`` generator class.
The associated simulation function is in ``sim_funcs/surmise_test_function.py``,
and the default ``only_persistent_gens`` allocation function is used automatically.

Using cancellations to kill running simulations
------------------------------------------------

If a generated point is cancelled by the generator before it has been given to a worker for evaluation,
then it will never be given. If it has already returned from the simulation, then results can be returned,
but the ``cancel_requested`` field remains as True. However, if the simulation is running when the manager
receives the cancellation request, a kill signal will be sent to the worker. This can be caught and acted upon
by a user function, otherwise it will be ignored. To demonstrate this, the test ``test_persistent_surmise_killsims.py``
captures and processes this signal from the manager.

In order to do this, a compiled version of the borehole function is launched by ``sim_funcs/borehole_kills.py``
via the :doc:`Executor<../executor/ex_index>`. As the borehole application used here is serial, we use the
:doc:`Executor base class<../executor/ex_index>` rather than the commonly used :doc:`MPIExecutor<../executor/ex_index>`
class. The base Executor submit routine simply sub-processes a serial application in-place. After the initial
sample batch of evaluations has been processed, an artificial delay is added to the sub-processed borehole to
allow time to receive the kill signal and terminate the application. Killed simulations will be reported at
the end of the test. As this is dependent on timing, the number of killed simulations will vary between runs.
This test is added simply to demonstrate the killing of running simulations and thus uses a reduced number of evaluations.

.. _surmise: https://github.com/mosesyhc/surmise
