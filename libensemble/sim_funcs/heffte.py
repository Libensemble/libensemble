"""
Example sim_f for simple heFFTe use case.
"""
import numpy as np
import subprocess


def call_and_process_heffte(H, persis_info, sim_specs, _):
    """
    Evaluates (via subprocess) a string that includes a call to a heFFTe
    executable as well as other arguments. Afterwards, the stdout is parsed to
    collect the run time (as reported by heFTTe)
    """

    H_o = np.zeros(1, dtype=sim_specs["out"])

    runstring = "mpirun -np 4 ./speed3d_c2c fftw double 128 128 128" + " " + str(H[0]['p0']) + " " + str(H[0]['p1']) + " " + str(H[0]['p2']) + " " + str(H[0]['p3'])

    p = subprocess.run(runstring.split(" "), cwd="./", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert p.returncode == 0, "heFFTe call has failed"

    time = float(p.stdout.decode().split("Time per run: ")[1].split(" ")[0])

    H_o["RUN_TIME"] = time
    return H_o, persis_info
