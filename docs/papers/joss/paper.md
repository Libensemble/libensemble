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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

<!-- The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration). -->

As the number of available computational resources increases, almost all applications
or evaluations evantually stop perfectly scaling. Nonetheless, clusters, servers, and other resources
keep growing, alongside the request to efficiently apply that hardware.
libEnsemble is a complete Python toolkit and workflow system for intelligently driving
"ensembles" of experiments or simulations at massive scales. Via a generator-simulator
(or decision/learner-evaluator) paradigm, libEnsemble enables and encourages multi-disciplinary
design, decision, and inference studies on or across laptops and heterogeneous hardware.

# Statement of need

Examples of way in which libEnsemble has been used in science and engineering problems include
- optimization of variational algorithms on quantum computers [@Liu2022layer]
- parallelize the ParMOO solver for multiobjective simulation optimization problems [@ParMOODesign23]
- design of particle accelerators [@Neveu2023]
- sequential Bayesian experimental design [@Surer2023] and Bayesian calibration [@MCMPSW2022]

Additional details on the parallel features and scalability of libEnsemble can be found in Refs [@Hudson2022] and [@libensemble-man].

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from David Bindel (Anyone else we want to ack? Moses and Tyler?)
This article was supported in part by the PETSc/TAO activity within the U.S. Department of Energy's (DOE's) Exascale Computing Project (17-SC-20-SC) and by the ComPASS and NUCLEI SciDAC projects within DOE's Office of Science, Advanced Scientific Computing Research under contract numbers DE-AC02-06CH11357 and DE-AC02-05CH11231.


# References
