import numpy as np
from sine_gen import gen_random_sample
from sine_sim import sim_find_sine

from libensemble import Ensemble
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    libE_specs = LibeSpecs(nworkers=4, comms="local")

    gen_specs = GenSpecs(
        gen_f=gen_random_sample,  # Our generator function
        out=[("x", float, (1,))],  # gen_f output (name, type, size)
        user={
            "lower": np.array([-6]),  # lower boundary for random sampling
            "upper": np.array([6]),  # upper boundary for random sampling
            "gen_batch_size": 10,  # number of x's gen_f generates per call
        },
    )

    sim_specs = SimSpecs(
        sim_f=sim_find_sine,  # Our simulator function
        inputs=["x"],  # InputArray field names. "x" from gen_f output
        out=[("y", float)],  # sim_f output. "y" = sine("x")
    )

    exit_criteria = ExitCriteria(gen_max=160)

    ensemble = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs)
    ensemble.add_random_streams()
    ensemble.run()

    if ensemble.flag != 0:
        print("Oh no! An error occurred!")
