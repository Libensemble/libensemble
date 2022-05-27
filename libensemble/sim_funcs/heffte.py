"""
Calls the branin function. Default behavior uses the python function, but
uncommenting lines will write x.in to file, call branin.py, and then read f.out.
"""
import numpy as np
import subprocess


def call_and_process_heffte(H, persis_info, sim_specs, _):
    """Evaluates a heffte string and parses the output"""

    H_o = np.zeros(1, dtype=sim_specs["out"])

    p = subprocess.run(H["exec_and_args"][0].split(" "), cwd="./", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time = float(p.stdout.decode().split("Time per run: ")[1].split(" ")[0])

    H_o["run_time"] = time
    return H_o, persis_info
