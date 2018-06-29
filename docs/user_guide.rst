libEnsemble User Guide
======================

libEnsemble overview
--------------------
libEnsemble is a software library to coordinate the concurrent evaluation of ensembles of calculations. libEnsemble uses a manager to allocate work to various workers. (A libEnsemble worker is the smallest indivisible unit to perform some calculation.) The work performed by libEnsemble is governed by three routines:

* gen_f: Generates inputs to sim_f.
* sim_f: Evaluates a simulation or other evaluation at output from gen_f.
* alloc_f: Decides whether sim_f or gen_f should be called (and with what input/resources) as workers become available.

Example sim_f, gen_f, and alloc_f routines can be found in the examples/sim_funcs, examples/gen_funcs, and examples/alloc_funcs directories, respectively. Examples of scripts used for calling libEnsemble can be found at examples/calling_scripts/. To enable portability, a job_controller interface is supplied for users to launch and monitor jobs in their user-provided sim_f and gen_f routines.

The default alloc_f tells each available worker to call sim_f with the highest priority unit of work from gen_f. If a worker is idle and there is no gen_f output to give, the worker is told to call gen_f.


Expected use cases
------------------

Below are some expected libEnsemble use cases that we support (or are working to support) and plan to have examples of:

* A user is looking to optimize a simulation calculation. The simulation may already be using
  parallel resources, but not a large fraction of some computer. libEnsemble can coordinate the concurrent evaluation of the simulation sim_f at various parameter values and gen_f would return candidate parameter values (possibly after each sim_f output).

* A user is has a gen_f that produces different meshes to be used within a sim_f. Given the sim_f
  output, gen_f will refine a mesh or produce a new mesh. libEnsemble can ensure that the calculated
  meshes can be used by multiple simulations without requiring movement of data.

* A user is attempting to sample a simulation sim_f at some parameter values, many of which are
  will cause the simulation to fail. libEnsemble can stop unresponsive evaluations, and recover com-
  putational resources for future evaluations. gen_f can possibly update the sampling after discovering regions where evaluations of sim_f fail.

* A user has a simulation sim_f that requires calculating multiple expensive quantities, some of
  which depend on other quantities. sim_f can observe intermediate quantities in order to stop related calcu-lations and preempt future calculations associated with a poor parameter values.

* A user has a simulation sim_f with multiple fidelities, with the higher-fidelity evaluations
  requiring more computational resources, and a gen_f/alloc_f that decides which parameters should be evalu-ated and at what fidelity level. libEnsemble can coordinate these evaluations without requiring the user know parallel programming.

* A user wishes to identify multiple local optima for a sim_f. Furthermore, sensitivity analysis is 
  desired at each identified optimum. libEnsemble can use the points from the APOSMM gen_f to identify optima and after a point is ruled to be an optimum, a different gen_f can produce a collection of parameters necessary for sensitivity analysis of sim_f.
  

Naturally, combinations of these use cases are supported as well. An example of such a combination is using libEnsemble to solve an optimization problem that relies on simulations that fail frequently.


The libEnsemble History Array
-----------------------------

libEnsemble uses a numpy structured array H to store output from gen_f and corresponding sim_f output. Similarly, gen_f and sim_f are expected to return output in numpy structured arrays. The names of the fields to be given as input to gen_f and sim_f must be an output from gen_f or sim_f. In addition to the fields output from sim_f and gen_f, the final history returned from libEnsemble will include the fields. (Note that the libEnsemble history can contain pointers to data instead of the data itself. In some applications, this can greatly the size of the history and reduce the amount of data communicated to/from the manager.):

* sim_id' [int]: Each unit of work output from gen_f is must have an associated sim_id. The generator
  can assign this, but users must careful to ensure points are added in order. For example, if alloc_f allows for two gen_f instances to be running simultaneously, alloc_f should ensure that both don’t generate points with the same sim_id.

* given' [bool]: Has this gen_f output been given to a libEnsemble worker to be evaluated yet?

* given_time' [float]: At what time (since the epoch) was this gen_f output given to a worker?

* sim_worker' [int]: libEnsemble worker that it was given to be evaluated.

* gen_worker' [int]: libEnsemble worker that generated this sim_id

* returned' [bool]: Has this a worker completed the evaluation of this unit of work?

* paused' [bool]: Has this evaluation been paused?











