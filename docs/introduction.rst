.. include:: ../README.rst
    :start-after: after_badges_rst_tag


Basic Usage
===========

.. basic_usage

Select or supply Simulator and Generator functions
--------------------------------------------------

**Generator** and **Simulator** Python functions respectively produce candidate parameters and
perform/monitor computations that use those parameters. Coupling them together with libEnsemble is easy:

.. code-block:: python

    from my_simulators import beamline_simulation_function
    from someones_calibrator import adaptive_calibrator_function

    from libensemble import Ensemble, SimSpecs, GenSpecs, LibeSpecs, ExitCriteria

    if __name__ == "__main__":

        basic_settings = LibeSpecs(comms="local", nworkers=16, save_every_k_gens=100, kill_cancelled_sims=True)

        simulation = SimSpecs(sim_f=beamline_simulation_function, inputs=["x"], out=[("f", float)])

        outer_loop = GenSpecs(gen_f=adaptive_calibrator_function, inputs=["f"], out=[("x", float)])

        when_to_stop = ExitCriteria(gen_max=500)

        my_experiment = Ensemble(basic_settings, simulation, outer_loop, when_to_stop)

        Output = my_experiment.run()


Launch and monitor apps on parallel resources
---------------------------------------------

libEnsemble includes an Executor interface so application-launching functions are
portable, resilient, and flexible. It automatically detects available resources and GPUs,
and can dynamically assign them:

.. code-block:: python

    import numpy as np
    from libensemble.executors import MPIExecutor


    def beamline_simulation_function(Input):

        particles = str(Input["x"])
        args = "timesteps " + str(10) + " " + particles

        exctr = MPIExecutor()
        exctr.register_app("./path/to/particles.app", app_name="particles")

        # GPUs selected by Generator, can autotune or set explicitly
        task = exctr.submit(app_name="particles", app_args=args, num_procs=64, auto_assign_gpus=True)

        task.wait()

        try:
            data = np.loadtxt("particles.stat")
            final_energy = data[-1]
        except Exception:
            final_energy = np.nan

        output = np.zeros(1, dtype=[("f", float)])
        output["energy"] = final_energy

        return output

See the `user guide`_ for more information.

.. example_packages

.. dropdown:: **Example Compatible Packages**

  libEnsemble and the `Community Examples repository`_ include example generator
  functions for the following libraries:

  - APOSMM_ Asynchronously parallel optimization solver for finding multiple minima. Supported local optimization routines include:

    - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
    - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
    - `scipy.optimize`_ Open-source solvers for nonlinear problems, linear programming,
      constrained and nonlinear least-squares, root finding, and curve fitting.
    - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

  - DEAP_ Distributed evolutionary algorithms
  - Distributed optimization methods for minimizing sums of convex functions. Methods include:

    - Primal-dual sliding (https://arxiv.org/pdf/2101.00143).
    - Distributed gradient descent with gradient tracking (https://arxiv.org/abs/1908.11444).
    - Proximal sliding (https://arxiv.org/abs/1406.0919).

  - ECNoise_ Estimating Computational Noise in Numerical Simulations
  - Surmise_ Modular Bayesian calibration/inference framework
  - Tasmanian_ Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN
  - VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization

  libEnsemble has also been used to coordinate many computationally expensive
  simulations. Select examples include:

  - OPAL_ Object Oriented Parallel Accelerator Library. (See this `IPAC manuscript`_.)
  - WarpX_ Advanced electromagnetic particle-in-cell code. (See example `WarpX + libE scripts`_.)


.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples

.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _NLopt: https://nlopt.readthedocs.io/en/latest/
.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _PETSc/TAO: http://www.mcs.anl.gov/petsc
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _Tasmanian: https://tasmanian.ornl.gov/
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _OPAL: http://amas.web.psi.ch/docs/opal/opal_user_guide-1.6.0.pdf
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _IPAC manuscript: https://doi.org/10.18429/JACoW-ICAP2018-SAPAF03
.. _WarpX + libE scripts: https://warpx.readthedocs.io/en/latest/usage/workflows/libensemble.html
