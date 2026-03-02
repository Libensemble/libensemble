import numpy as np
from gest_api.vocs import VOCS
from sine_gen_std import RandomSample
from sine_sim import sim_find_sine

from libensemble import Ensemble
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":  # Python-quirk required on macOS and windows
    libE_specs = LibeSpecs(nworkers=4, comms="local")

    vocs = VOCS(variables={"x": [-3, 3]}, objectives={"y": "EXPLORE"})  # Configure our generator with this object

    generator = RandomSample(vocs)  # Instantiate our generator

    gen_specs = GenSpecs(
        generator=generator,  # Pass our generator and config to libEnsemble
        vocs=vocs,
        batch_size=4,
    )

    sim_specs = SimSpecs(
        sim_f=sim_find_sine,  # Our simulator function
        inputs=["x"],  # InputArray field names. "x" from gen_f output
        out=[("y", float)],  # sim_f output. "y" = sine("x")
    )  # sim_specs_end_tag

    exit_criteria = ExitCriteria(sim_max=80)  # Stop libEnsemble after 80 simulations

    ensemble = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs)
    ensemble.alloc_specs = AllocSpecs(alloc_f=give_sim_work_first)
    ensemble.add_random_streams()  # setup the random streams unique to each worker
    ensemble.run()  # start the ensemble. Blocks until completion.

    history = ensemble.H  # start visualizing our results

    print([i for i in history.dtype.fields])  # (optional) to visualize our history array
    print(history)

    import matplotlib.pyplot as plt

    colors = ["b", "g", "r", "y", "m", "c", "k", "w"]

    for i in range(1, libE_specs.nworkers + 1):
        worker_xy = np.extract(history["sim_worker"] == i, history)
        x = [entry.tolist()[0] for entry in worker_xy["x"]]
        y = [entry for entry in worker_xy["y"]]
        plt.scatter(x, y, label="Worker {}".format(i), c=colors[i - 1])

    plt.title("Sine calculations for a uniformly sampled random distribution")
    plt.xlabel("x")
    plt.ylabel("sine(x)")
    plt.legend(loc="lower right")
    plt.savefig("tutorial_sines.png")
