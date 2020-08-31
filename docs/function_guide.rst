==================================
Writing libEnsemble User Functions
==================================

libEnsemble coordinates ensembles of calculations performed by three
main functions: a :ref:`Generator Function<api_gen_f>`, a
:ref:`Simulator Function<api_sim_f>`, and an :ref:`Allocation Functions<api_alloc_f>`,
or ``gen_f``, ``sim_f``, and ``alloc_f`` respectively. These are all referred to
as User Functions. Although libEnsemble includes several ready-to-use User Functions
like :doc:`APOSMM<examples/aposmm>`, it's expected that most users will write their own.
This guide serves as an overview of both necessary and optional components for
writing different kinds of User Functions, and common development patterns.

Generator Function
==================

As described in the :ref:`API<api_gen_f>`, the ``gen_f`` is called by libEnsemble
via the following::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

In practice, most ``gen_f`` function definitions resemble::

    def my_generator(H, persis_info, gen_specs, _):

Where :doc:`H<data_structures/history_array>` is a selection of the
:doc:`History array<history_output>`, determined by sim IDs from the
``alloc_f``, and :doc:`persis_info<data_structures/persis_info>` is a dictionary
containing state information. See the API above for detailed descriptions of the other parameters.

Typically users start by parsing their custom parameters initially defined
within ``gen_specs['user']`` in the calling script and defining a *local* History
array based on the datatype in ``gen_specs['out']``, to be returned. For example::

        batch_size = gen_specs['user']['batch_size']
        local_H_out = np.zeros(batch_size, dtype=gen_specs['out'])

This array should be populated by whatever values are generated within
the function. Finally, this array should be returned to libEnsemble
alongside ``persis_info``::

        return local_H_out, persis_info

Persistent Generator
--------------------

While normal generators return after completing their calculation, persistent
generators receive Work units and communicate results directly to the manager
in a loop, often not returning until the conclusion of the entire libEnsemble
routine. The calling worker becomes a dedicated
:ref:`persistent worker<persis_worker>`.  The ``gen_f`` is initiated as
persistent by the ``alloc_f``.

Simulator Function
==================

Executor
--------

Allocation Function
===================
