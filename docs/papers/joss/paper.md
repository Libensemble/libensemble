---
title: 'libEnsemble: A complete toolkit for dynamic ensembles of calculations'
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
date: 13 October 2023
bibliography: paper.bib
---

# Summary

As the number of available computational resources increases, almost all applications
or evaluations eventually stop scaling. Nonetheless, clusters, servers, and other resources
keep growing, alongside the request to efficiently apply that hardware.
libEnsemble is a complete Python toolkit and workflow system for intelligently driving
"ensembles" of experiments or simulations at massive scales. Via a generator-simulator
(or decision/learner-evaluator) paradigm, libEnsemble enables and encourages multi-disciplinary
design, decision, and inference studies on or across laptops and heterogeneous hardware.

# Statement of need

There are a growing number of packages aimed at workflows and a sub-set of these focus on running ensembles of calculations on clusters and supercomputers. A dynamic ensemble refers to packages that automatically steer the ensemble based on intermediate results. This may involve deciding simulation parameters based on numerical optimization or machine learning techniques, among other possibilities. Other packages in this space include Colmena [@colmena21] and the RADICAL-Ensemble Toolkit [@ensembletoolkit16] and also packages that provide back-end dispatch and execution such as Parsl [@parsl] and Balsam [@Salim2019].

Some crucial considerations relevant to these packages include:

- Ease of use - whether the software requires a complex setup.

- Portability - running on different machines with different schedulers, hardware, and MPI runners with minimal modification to user scripts.

- Scalability - working efficiently with large and/or many concurrent simulations.

- Interoperability - the modularity of the package and the ability to interoperate with other packages.

- Dynamic resources - the ability to dynamically assign machine resources to simulations.

- Ability to cancel simulations on the fly.

[merge sim/gen with this?]

LibEnsemble stands out primarily through its generator-simulator paradigm, which eliminates the need for users to explicitly define task dependencies. Instead, it emphasizes data dependencies between customizable Python user functions. This modular design also lends itself to exploiting the large library of example user functions that are provided with libEnsemble or available from the community, maximizing code reuse. For instance, users can readily choose an existing generator function and tailor a simulator function to their particular needs.

<!--- JLN: can reword the below paragraph -->

libEnsemble is a complete toolkit that includes generator in-the-loop and backend mechanisms. Some other packages cover a sub-set of the workflow. Colmena, for example, has a front-end that uses components to create and coordinate tasks while using Parsl to dispatch simulations. 

libEnsemble communicates between a manager and multiple workers using either Python's built-in multiprocessing, MPI (via mpi4py), or TCP.

libEnsemble takes the philosophy of minimizing required dependencies while supporting various back-end mechanisms when needed.

For example, the vast majority of use-cases do not require a database or special run-time. but for those that do, Balsam can be used on the back-end by substituting the regular MPI executor for the Balsam executor. This approach simplifies the user experience and reduces the initial setup and adoption costs when using libEnsemble.

To achieve portability, libEnsemble employs system detection beyond other packages. It detects crucial system information such as scheduler details, MPI runners, core counts, and GPU counts (for different types of GPU) and uses these to produce run-lines and GPU settings for these systems, without the user having to alter scripts. For example, on a system using "srun", libEnsemble will use srun options to assign GPUs, while on other systems it may assign via environment variables such as ROCR_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, while the user only states the number of GPUs needed for each simulation. For cases where autodetection is insufficient, the user can supply platform information or the name of a known system via scripts or an environment variable.

By default, libEnsemble divides available compute resources amongst workers. However, when simulation parameters are created, the number of processes and GPUs can also be specified for each simulation. Combined with the portability features, this makes it very simple to transfer user scripts between platforms.

The close coupling between the libEnsemble generator and simulators enables the generator to perform tasks such as asynchronously receiving results, updating models, and canceling previously initiated simulations. Simulations that are already running can be terminated and resources recovered. This is more flexible compared to other packages, where the generation of simulations is external to the dispatch of a batch of simulations.

libEnsemble also supports persistent user functions that run on workers, maintaining their memory, which prevents the storing and reloading of data required by packages that only support a fire-and-forget approach to ensemble components.

# Example use-cases

Examples of ways in which libEnsemble has been used in science and engineering problems include:

- Optimization of variational algorithms on quantum computers [@Liu2022layer].
- Parallelization of the ParMOO solver for multiobjective simulation optimization problems [@ParMOODesign23].
- Design of particle accelerators [@Neveu2023] [@PhysRevAccelBeams.26.084601] [@Pousa22].
- Sequential Bayesian experimental design [@Surer2023] and Bayesian calibration [@MCMPSW2022].

libEnsemble's generators and simulators, commonly referred to as user functions, are Python
functions that simply accept and return NumPy structured arrays. Generators produce input for
simulators, while simulators evaluate those inputs; any level of complexity is supported.
Multiple concurrent instances ("ensembles") of user functions are coordinated by libEnsemble's
worker processes. Workers are typically assigned/reassigned compute resources, and within
user functions can launch applications, evaluate intermediate results, and statefully intercommunicate.

Additional details on the parallel features and scalability of libEnsemble can be found in Refs [@Hudson2022] and [@libensemble-man].

# Acknowledgements

We acknowledge contributions from David Bindel.
This article was supported in part by the PETSc/TAO activity within the U.S. Department of Energy's (DOE's) Exascale Computing Project (17-SC-20-SC) and by the CAMPA, ComPASS, and NUCLEI SciDAC projects within DOE's Office of Science, Advanced Scientific Computing Research under contract numbers DE-AC02-06CH11357 and DE-AC02-05CH11231.


# References
