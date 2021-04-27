======================
Writing User Functions
======================

An early part of libEnsemble's design was the decision to divide ensemble steps into
generator and simulator routines as an intuitive way to express problems and their inherent
dependencies.

libEnsemble was consequently developed to coordinate ensemble computations defined by

• a *generator function* that produces simulation inputs,
• a *simulator function* that performs and monitors simulations, and
• an *allocation function* that determines when (and with what resources) the other two functions should be invoked.

Since each of these functions is supplied or selected by libEnsemble's users, they are
typically referred to as *user functions*. User functions need not be written only in
Python: they can (and often do) depend on routines from other
languages. The only restriction for user functions is that their inputs and outputs conform
to the :ref:`user function APIs<user_api>`. Therefore, the level of computation and complexity of any user function
can vary dramatically based on the user's needs.

These guides describe common development
patterns and optional components for each kind of User Function:

.. toctree::
   :maxdepth: 2

   generator
   simulator
   allocator
