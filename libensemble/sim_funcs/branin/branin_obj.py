"""
Calls the branin function. Default behavior uses the python function, but
uncommenting lines will write x.in to file, call branin.py, and then read f.out.
"""
import numpy as np
import time
from libensemble.sim_funcs.branin.branin import branin


def call_branin(H, persis_info, sim_specs, _):
    """Evaluates the Branin function"""
    batch = len(H["x"])

    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        # Uncomment the following if you want to use the file system to do evaluations
        # devnull = open(os.devnull, 'w')
        # np.savetxt('./x.in', x, fmt='%16.16f', delimiter=' ', newline=" ")
        # p = subprocess.call(['python', 'branin.py'], cwd='./', stdout=devnull)
        # H_o['f'][i] = np.loadtxt('./f.out', dtype=float)

        H_o["f"][i] = branin(x[0], x[1])

        if "user" in sim_specs and "uniform_random_pause_ub" in sim_specs["user"]:
            time.sleep(sim_specs["user"]["uniform_random_pause_ub"] * np.random.uniform())

    return H_o, persis_info
