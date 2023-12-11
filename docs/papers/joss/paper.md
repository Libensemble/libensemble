---
title: 'libEnsemble: A complete Python toolkit for dynamic ensembles of calculations'
tags:
  - Python
  - ensemble workflows
  - optimization and learning
authors:
  - name: Stephen Hudson
    orcid: 0000-0002-7500-6138
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Jeffrey Larson
    orcid: 0000-0001-9924-2082
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: John-Luke Navarro
    orcid: 0000-0002-9916-9038
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Stefan M. Wild
    orcid: 0000-0002-6099-2772
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: Mathematics and Computer Science Division, Argonne National Laboratory, Lemont, IL, USA
   index: 1
 - name: AMCR Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
   index: 2
 - name: Industrial Engineering and Management Sciences, Northwestern University, Evanston, IL, USA
   index: 3
date: 31 October 2023
bibliography: paper.bib
---

# Summary

Almost all science and engineering applications eventually stop scaling: their runtime no
longer decreases as available computational resources increase.
Therefore, many applications will struggle to efficiently use emerging
extreme-scale high-performance, parallel, and distributed systems.
libEnsemble is a complete Python toolkit and workflow system for intelligently driving
*ensembles* of experiments or simulations at massive scales.
It enables and encourages multidisciplinary design, decision, and inference
studies portably running on laptops, clusters, and supercomputers.

# Statement of Need

While a growing number of packages are aimed at workflows, relatively few
focus on running dynamic ensembles of calculations on clusters and supercomputers.
Dynamic ensembles are workflows of computations that are defined and steered
based on intermediate results.
Examples include determining simulation parameters using numerical optimization
methods, machine learning techniques, or statistical calibration tools. In each of
these examples, the ensemble members are typically simulations that use different
parameters or data. Additional examples of applications that have used libEnsemble are
surveyed in [Representative libEnsemble Use Cases](#Representative-libEnsemble-Use-Cases).

The target audience for libEnsemble includes scientists, engineers, and other researchers
who stand to benefit from such dynamic workflows.

Key considerations for packages running dynamic ensembles include the following:

- Ease of use -- whether the software requires a complex setup

- Portability -- running on diverse machines with different schedulers, hardware, and communication modes (e.g., MPI runners) with minimal modification to user scripts

- Scalability -- working efficiently with large-scale and/or many concurrent simulations

- Interoperability -- the modularity of the package and the ability to interoperate with other packages

- Adaptive resource management -- the ability to adjust resources given to each simulation throughout the ensemble

- Efficient resource utilization -- including the ability to cancel simulations on the fly

libEnsemble seeks to satisfy the above criteria using a generator--simulator--allocator
model. libEnsemble's generators, simulators, and allocators -- commonly referred to as
user functions -- are simply Python
functions that accept and return NumPy [@harris2020array] structured arrays. Generators produce input for
simulators, simulators evaluate those inputs, and allocators decide whether and when
a simulator or generator should be called; any level of complexity is supported.
Multiple concurrent instances (an "ensemble") of user functions are coordinated by libEnsemble's
worker processes. Workers are typically assigned/reassigned compute resources; within
user functions, workers can launch applications, evaluate intermediate results,
and communicate via the manager.

## Related Work

Other packages for managing workflows and ensembles include Colmena [@colmena21] and the
RADICAL-Ensemble Toolkit [@ensembletoolkit16] as well as packages such as Parsl
[@parsl] and Balsam [@Salim2019] that provide backend dispatch and execution.

libEnsemble's unique generator--simulator--allocator
paradigm eliminates the need for users to explicitly define task dependencies.
Instead, it emphasizes data dependencies between these customizable Python user
functions. This modular design also lends itself to exploiting the large
library of example user functions provided with libEnsemble or
available from the community (e.g., [@libEnsembleCommunityExamples]),
maximizing code reuse. For instance, users can
readily choose an existing generator function and tailor a simulator function
to their particular needs.

libEnsemble takes the philosophy of minimizing required dependencies while
supporting various backend mechanisms when needed.
In contrast to other packages that cover only a
subset of such a workflow,
libEnsemble is a complete toolkit that includes generator-in-the-loop and
backend mechanisms.
For example, Colmena uses frontend components to create and
coordinate tasks while using Parsl to dispatch simulations.

For example, the vast majority of current use cases do not require a database or
special runtime environment. For use cases that have such requirements, Balsam
can be used on the backend by
substituting the regular MPI executor for the Balsam executor. This approach
simplifies the user experience and reduces the initial setup and adoption costs
when using libEnsemble.

## libEnsemble Functionality

libEnsemble communicates between a manager and multiple workers using either
Python's built-in multiprocessing, MPI (via mpi4py [@Dalcin2008]), or TCP.

To achieve portability, libEnsemble detects runtime setup information not
commonly detected by other packages:
It detects crucial system information such as scheduler details, MPI
runners, core counts, and GPU counts (for different types of GPUs) and uses
these to produce run-lines and GPU settings for these systems, without the user
having to alter scripts. For example, on a system that uses Slurm's `srun`, libEnsemble
will use `srun` options to assign GPUs, while on other systems it will assign
GPUs via the appropriate environment variables such as `ROCR_VISIBLE_DEVICES`
or `CUDA_VISIBLE_DEVICES`,
allowing the user to simply state the number of GPUs needed for each simulation. For
cases where autodetection is insufficient, the user can supply platform
information or the name of a known system via scripts or an environment
variable. This makes it simple to transfer user scripts between platforms.


By default, libEnsemble equally divides available compute resources among workers.
When simulation parameters are created, however, the number of processes and
GPUs can also be specified for each simulation.
The close coupling between the libEnsemble generators and simulators enables a
generator to perform tasks such as asynchronously receiving results, updating
models, and canceling previously initiated simulations. Simulations that are
already running can be terminated and resources recovered. This approach is more
flexible compared with other packages, where the generation of simulations is
external to the dispatch of a batch of simulations.

libEnsemble also supports "persistent user functions" that
maintain their state while running on workers. This prevents the need to
store and reload data
as is done by other ensemble packages that support only a fire-and-forget
approach to ensemble components.

# Representative libEnsemble Use Cases

Examples of libEnsemble applications in science and engineering include the following:

- Optimization of variational algorithms on quantum computers [@Liu2022layer]
- Parallelization of the ParMOO solver for multiobjective simulation optimization problems [@ParMOODesign23]
- Design of particle accelerators [@Neveu2023] [@PhysRevAccelBeams.26.084601] [@Pousa22]
- Sequential Bayesian experimental design [@Surer2023] and Bayesian calibration [@MCMPSW2022]

A selection of community-provided libEnsemble functions and workflows that
users can build off is maintained in [@libEnsembleCommunityExamples].

Additional details on the parallel features and scalability of libEnsemble can
be found in [@Hudson2022] and [@libensemble-man].

# Acknowledgments

We acknowledge contributions from David Bindel.
This article was supported in part by the PETSc/TAO activity within the U.S.
Department of Energy's (DOE's) Exascale Computing Project (17-SC-20-SC) and by
the CAMPA, ComPASS, and NUCLEI SciDAC projects within DOE's Office of Science,
Advanced Scientific Computing Research under contract numbers DE-AC02-06CH11357
and DE-AC02-05CH11231.


# References
