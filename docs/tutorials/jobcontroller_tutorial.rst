=======================
Job Controller Tutorial
=======================

This tutorial demonstrates libEnsemble's capability to launch and monitor external
scripts or user applications within simulation or generator functions using the
 :ref:`Job Controller<jobcontroller_index>`.

While most high-performance systems contain some mechanism like ``jsrun`` or
 ``aprun`` for submitting applications for processing, hardcoding these routines
 as-is into a ``sim_f`` isn't a portable solution. See the Job Controller docs
 above for an in-depth overview of the Job Controller's capabilities.
