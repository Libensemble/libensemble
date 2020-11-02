======================================================
Selective Pending Sim Cancellation with Persistent CWP
======================================================

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on asynchronous directives from the *Persistent CWP* calibration
generator function. This capability is critical for this calibration use-case since
it isn't useful for the generator to receive pending, extraneous evaluations
from resources that may be more effectively applied towards possibly critical
evaluations.

[BETTER JUSTIFICATION GOES HERE?]

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings and components within the calling script and CWP
:ref:`persistent generator<persistent-gens>` that make this capability possible,
rather than outlining a step-by-step process for writing this exact use-case.
Nonetheless, we hope that these selections are inspirational for implementing
similar approaches in other user functions.

Persistent Generator Function
-----------------------------

Like all generator functions, a local :doc:`History array<../history_array>`
is initialized with a given size and datatype::

    H_o = np.zeros(pre_count, dtype=gen_specs['out'])

``gen_specs['out']`` is defined in our calling script, like all specification
dictionaries explored in previous tutorials. For now, know that ``gen_specs['out']``
contains the following two elements (alongside many others)::

    [('cancel', bool), ('kill_sent', bool)]
