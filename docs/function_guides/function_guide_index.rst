======================
Writing User Functions
======================

An early part of libEnsemble's design was the decision to divide ensemble steps into
generator and simulator routines as an intuitive way to express problems and their inherent
dependencies.

libEnsemble was consequently developed to coordinate ensemble computations defined by

• a **generator function** that produces simulation inputs,
• a **simulator function** that performs and monitors simulations, and
• an **allocation function** that determines when (and with what resources) the other two functions should be invoked.

Since each of these functions is supplied or selected by libEnsemble's users, they are
typically referred to as **user functions**. User functions typically only require a
basic familiarity with NumPy_, but as long as they conform to
the :ref:`user function APIs<user_api>`, they can incorporate any other machine-learning,
mathematics, resource management, or other libraries/applications. Therefore, the
level of computation and complexity of any user function can vary dramatically based
on the user's needs.

These guides describe common development
patterns and optional components for each kind of User Function:

.. toctree::
   :maxdepth: 2

   generator
   simulator
   allocator

.. _NumPy: http://www.numpy.org
