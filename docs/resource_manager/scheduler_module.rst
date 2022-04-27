.. _resources-scheduler:

Scheduler Module
================

The scheduler is called within the scope of the allocation function, usually
via the ``alloc_support`` module function ``assign_resources()`` (either
called directly or via ``sim_work()`` or ``gen_work()``), which
is a wrapper for the main scheduler function ``assign_resources()``.

The alloc_support module allows users to supply an alternative scheduler
that fits this interface. This could be achieved, for example, by inheriting
the built-in scheduler and making modifications.

Options can also be provided to the scheduler though the
``libE_specs['scheduler_opts']`` dictionary.

.. automodule:: resources.scheduler

.. autoclass:: ResourceScheduler
  :member-order: bysource
  :members: __init__, assign_resources
