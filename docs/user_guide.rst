Introduction
============

libEnsemble overview
--------------------
libEnsemble is a software library to coordinate the concurrent evaluation of
ensembles of calculations. libEnsemble uses a manager to allocate work to
various workers. (A libEnsemble worker is the smallest indivisible unit to
perform some calculation.) The work performed by libEnsemble is governed by
three routines:

* gen_f: Generates inputs to sim_f.
* sim_f: Evaluates a simulation or other evaluation at output from gen_f.
* alloc_f: Decides whether sim_f or gen_f should be called (and with what input/resources) as workers become available.

Example sim_f, gen_f, and alloc_f routines can be found in the
``examples/sim_funcs/``, ``examples/gen_funcs/``, and ``examples/alloc_funcs/`` directories,
respectively. Examples of scripts used for calling libEnsemble can be found in
``examples/calling_scripts/``. To enable portability, a :doc:`job_controller<job_controller/overview>` 
interface is supplied for users to launch and monitor jobs in their user-provided sim_f and
gen_f routines.

The default alloc_f tells each available worker to call sim_f with the highest
priority unit of work from gen_f. If a worker is idle and there is no gen_f
output to give, the worker is told to call gen_f.


Expected use cases
------------------

Below are some expected libEnsemble use cases that we support (or are working
to support) and plan to have examples of:

* A user is looking to optimize a simulation calculation. The simulation may
  already be using parallel resources, but not a large fraction of some
  computer. libEnsemble can coordinate the concurrent evaluation of the
  simulation sim_f at various parameter values and gen_f would return candidate
  parameter values (possibly after each sim_f output).

* A user has a gen_f that produces different meshes to be used within a
  sim_f. Given the sim_f output, gen_f will refine a mesh or produce a new
  mesh. libEnsemble can ensure that the calculated meshes can be used by
  multiple simulations without requiring movement of data.

* A user is attempting to sample a simulation sim_f at some parameter values,
  many of which will cause the simulation to fail. libEnsemble can stop
  unresponsive evaluations, and recover computational resources for future
  evaluations. gen_f can possibly update the sampling after discovering regions
  where evaluations of sim_f fail.

* A user has a simulation sim_f that requires calculating multiple expensive
  quantities, some of which depend on other quantities. sim_f can observe
  intermediate quantities in order to stop related calculations and preempt
  future calculations associated with poor parameter values.

* A user has a sim_f with multiple fidelities, with the
  higher-fidelity evaluations requiring more computational resources, and a
  gen_f/alloc_f that decides which parameters should be evaluated and at what
  fidelity level. libEnsemble can coordinate these evaluations without
  requiring the user know parallel programming.

* A user wishes to identify multiple local optima for a sim_f. Furthermore,
  sensitivity analysis is desired at each identified optimum. libEnsemble can
  use the points from the APOSMM gen_f to identify optima; and after a point is
  ruled to be an optimum, a different gen_f can produce a collection of
  parameters necessary for sensitivity analysis of sim_f.


Naturally, combinations of these use cases are supported as well. An example of
such a combination is using libEnsemble to solve an optimization problem that
relies on simulations that fail frequently.


The libEnsemble History Array
-----------------------------

libEnsemble uses a numpy structured array H to store output from gen_f and
corresponding sim_f output. Similarly, gen_f and sim_f are expected to return
output in numpy structured arrays. The names of the fields to be given as input
to gen_f and sim_f must be an output from gen_f or sim_f. In addition to the
fields output from sim_f and gen_f, the final history returned from libEnsemble
will include the following fields:

* sim_id' [int]: Each unit of work output from gen_f must have an associated
  sim_id. The generator can assign this, but users must be careful to ensure
  points are added in order. For example, if alloc_f allows for two gen_f
  instances to be running simultaneously, alloc_f should ensure that both don’t
  generate points with the same sim_id.

* given' [bool]: Has this gen_f output been given to a libEnsemble worker to be
  evaluated yet?

* given_time' [float]: At what time (since the epoch) was this gen_f output
  given to a worker?

* sim_worker' [int]: libEnsemble worker that it was given to be evaluated.

* gen_worker' [int]: libEnsemble worker that generated this sim_id

* gen_time' [float]: At what time (since the epoch) was this entry (or
  collection of entries) put into H by the manager

* returned' [bool]: Has this worker completed the evaluation of this unit of
  work?


LibEnsemble Output
------------------

The history array is returned to the user by libEnsemble. In the case that libEnsemble
aborts on an exception, the existing history array is dumped to a file libE_history_at_abort_<sim_count>.npy, where sim_count is the number of points evaluated.

Other libEnsemble files produced by default are:

**libE_stats.txt**: This contains a one-line summary of all user calculations. Each calculation summary is sent by workers to the manager and printed as the run progresses.

**ensemble.log**: This is the logging output from libEnsemble. The default logging is at INFO level. To gain additional diagnostics logging level can be set to DEBUG. If this file is not removed, multiple runs will append output.
For more info, see :doc:`Logging<logging>`.
