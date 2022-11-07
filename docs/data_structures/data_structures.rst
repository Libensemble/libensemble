Configuring libEnsemble
=======================

This section outlines the dictionaries, classes, and other structures used by libEnsemble
to configure a workflow, oftentimes within a single Python file we refer to as a *calling script*.

.. We first discuss
.. the dictionaries that are input to libEnsemble to declare the
.. :ref:`simulation<datastruct-sim-specs>`,
.. :ref:`generator<datastruct-gen-specs>`,
.. and
.. :ref:`allocation<datastruct-alloc-specs>`
.. specifications, as well as specify
.. :ref:`exit criteria<datastruct-exit-criteria>`,
.. :ref:`persistent information<datastruct-persis-info>`, and other
.. :ref:`libEnsemble<datastruct-libe-specs>`
.. options.

.. We then discuss internal libEnsemble, including the
.. :ref:`history array<datastruct-history-array>`,
.. :ref:`worker array<datastruct-history-array>`,
.. and the
.. :ref:`work<datastruct-history-array>` dictionary produced by the allocation
.. function.


.. toctree::
   :maxdepth: 3
   :caption: libEnsemble Specifications:

   sim_specs
   gen_specs
   libE_specs
   alloc_specs
   persis_info
   exit_criteria
   ensemble_specs