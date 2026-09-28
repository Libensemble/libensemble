import numpy as np
from sine_gen import gen_random_sample
from sine_sim import sim_find_sine

from libensemble import Ensemble
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.specs import AllocSpecs, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    libE_specs = LibeSpecs(nworkers=4, comms="local")

    gen_specs = GenSpecs(
        gen_f=gen_random_sample,  # Our generator function
        out=[("x", float, (1,))],  # gen_f output (name, type, size)
        batch_size=10,  # number of x's gen_f generates per call
        user={
            "lower": np.array([-6]),  # lower boundary for random sampling
            "upper": np.array([6]),  # upper boundary for random sampling
        },
    )

    sim_specs = SimSpecs(
        sim_f=sim_find_sine,  # Our simulator function
        inputs=["x"],  # InputArray field names. "x" from gen_f output
        out=[("y", float)],  # sim_f output. "y" = sine("x")
    )

    alloc_specs = AllocSpecs(alloc_f=give_sim_work_first)

    ensemble = Ensemble(sim_specs, gen_specs, libE_specs=libE_specs, alloc_specs=alloc_specs)
    ensemble.run(gen_max=160)

    if ensemble.flag != 0:
        print("Oh no! An error occurred!")
