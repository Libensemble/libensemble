.. include:: ../README.rst
    :start-after: after_badges_rst_tag


Basic Usage
===========

Select or supply Simulator and Generator functions
--------------------------------------------------

**Generator** and **Simulator** Python functions respectively produce candidate parameters and
perform/monitor computations that use those parameters. Coupling them together with libEnsemble is easy::

    from my_simulators import beamline_simulation_function
    from someones_calibrator import adaptive_calibrator_function

    from libensemble import Ensemble, SimSpecs, GenSpecs, LibeSpecs, ExitCriteria

    if __name__ == "__main__":

        basic_settings = LibeSpecs(
          comms = "local",
          nworkers = 16,
          save_every_k_gens = 100,
          kill_cancelled_sims = True
        )

        simulation = SimSpecs(
          sim_f = beamline_simulation_function,
          inputs = ["x"],
          out = [("f", float)]
        )

        outer_loop = GenSpecs(
          gen_f = adaptive_calibrator_function,
          inputs = ["f"],
          out = [("x", float)]
        )

        when_to_stop = ExitCriteria(gen_max = 500)

        my_experiment = Ensemble(basic_settings, simulation, outer_loop, when_to_stop)

        Output = my_experiment.run()

Launch and monitor apps on parallel resources
---------------------------------------------

libEnsemble includes an Executor interface so application-launching functions are
portable, resilient, and flexible. It automatically detects available resources and GPUs,
and can dynamically assign them::

    import numpy as np
    from libensemble.executors import MPIExecutor

    def beamline_simulation_function(Input):

        particles = str(Input["x"])
        args = "timesteps " + str(10) + " " + particles

        exctr = MPIExecutor()
        exctr.register_app("./path/to/particles.app", app_name="particles")

        # GPUs selected by Generator, can autotune or set explicitly
        task = exctr.submit(app_name="particles", app_args=args,
                            num_procs=64, auto_assign_gpus=True)

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
