.. _datastruct-history-array:

history array
=============
::

    H: numpy structured array
        History to store output from gen_f/sim_f/alloc_f for each entry

libEnsemble uses a NumPy structured array :ref:`H<datastruct-history-array>` to
store output returned from each ``gen_f`` and ``sim_f``. Therefore,
``gen_f`` and ``sim_f`` are expected to return output as NumPy structured
arrays. The names of the input fields for ``gen_f`` and ``sim_f``
must be output from ``gen_f`` or ``sim_f``. In addition to output fields,
the final history array returned by libEnsemble will include the following *protected
fields*, populated internally:

* ``sim_id`` [int]: Each unit of work output from ``gen_f`` must have an
  associated ``sim_id``. The generator can assign this, but users must be
  careful to ensure that points are added in order. For example, if ``alloc_f``
  allows for two ``gen_f`` instances to be running simultaneously, ``alloc_f``
  should ensure that both don’t generate points with the same ``sim_id``.

* ``given`` [bool]: Has this ``gen_f`` output been given to a libEnsemble
  worker to be evaluated by a ``sim_f``?

* ``given_time`` [float]: At what time (since the epoch) was this ``gen_f``
  output *first* given to a worker to be evaluated by a ``sim_f``?

* ``last_given_time`` [float]: At what time (since the epoch) was this ``gen_f``
  output *last* given to a worker to be evaluated by a ``sim_f``?

* ``sim_worker`` [int]: libEnsemble worker that ``gen_f`` output was given to for evaluation

* ``gen_worker`` [int]: libEnsemble worker that (first) generated this ``sim_id``

* ``gen_time`` [float]: At what time (since the epoch) was this entry (or collection of entries) first put into ``H`` by the manager?

* ``last_gen_time`` [float]: At what time (since the epoch) was this entry last requested by a ``gen_f``?

* ``returned`` [bool]: Has this entry been evaluated by a ``sim_f``?

* ``returned_time`` [float]: At what time (since the epoch) was this entry *last* returned from a ``sim_f``?

* ``cancel_requested`` [bool]: Has cancellation of evaluation of this entry been requested?

* ``kill_sent`` [bool]: Has a kill signal been sent to the worker evaluating this entry?

Other than ``'sim_id'`` and ``cancel_requested``, protected fields cannot be
overwritten by user functions unless ``libE_specs['safe_mode']`` is set to ``False``.
