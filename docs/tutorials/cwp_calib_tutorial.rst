======================================================
Selective Pending Sim Cancellation with Persistent CWP
======================================================

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on asynchronous directives from the *Persistent CWP* calibration
generator function. This capability is critical for this calibration use-case since
it isn't useful for the generator to receive pending, extraneous evaluations
from resources that may be more effectively applied towards critical evaluations.

[BETTER JUSTIFICATION GOES HERE?]

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings, components, and data fields within the calling script and CWP
:ref:`persistent generator<persistent-gens>` that make this capability possible,
rather than outlining a step-by-step process for writing this exact use-case.
Nonetheless, we hope that these selections are inspirational for implementing
similar approaches in other user functions.

History Array Reserved Fields
-----------------------------

Like all generator functions, a local :doc:`History array<../history_array>`
is initialized with a given size and datatype::

    H_o = np.zeros(pre_count, dtype=gen_specs['out'])

``gen_specs['out']`` is defined in our calling script, like all specification
dictionaries explored in previous tutorials. For now, know that ``gen_specs['out']``
contains the following elements (alongside many others)::

    [('sim_id', int), ('cancel', bool), ('kill_sent', bool)]

``'sim_id'`` should be familiar from previous tutorials. While the CWP persistent
generator loops, it detects if any pending points (thetas) distributed for simulation
ought to be cancelled using a library function, then calls ``cancel_row()``::

    r_obviate = obviate_pend_thetas(model, theta, data_status)
    if r_obviate[0].shape[0] > 0:
        cancel_row(pre_count, r_obviate, n_x, data_status, comm)

Within ``cancel_row()``,
