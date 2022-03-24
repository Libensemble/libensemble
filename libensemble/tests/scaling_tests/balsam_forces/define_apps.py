from balsam.api import ApplicationDefinition

"""
This script uses the Balsam API to define and sync two types of Balsam apps:
a libEnsemble app, and a Forces app:

    - The libEnsemble app runs the calling script ``run_libe_forces_balsam.py``.
      An input transfer is also specified, but parameterized in
      ``submit_libe_forces_balsam.py`` as part of the Job specification process.

    - The Forces app is defined and synced with Balsam. The libEnsemble app
      will submit instances of the Forces app to the Balsam service for scheduling
      on a running batch session at its site. An optional output transfer is defined;
      forces.stat files are transferred back to the Globus endpoint defined in
      run_libe_forces_balsam.py

Unless changes are made to these Apps, this should only need to be run once to
register each of these apps with the Balsam service.
"""


class LibensembleApp(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro/.conda/envs/again/bin/python /home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/balsam_forces/run_libe_forces_balsam.py"
        + " > libe_out.txt 2>&1"
    )

    transfers = {
        "input_file": {
            "required": True,
            "direction": "in",
            "local_path": ".",
            "description": "Transfer in of balsam_forces.yaml",
            "recursive": False,
        }
    }

print("Defined LibensembleApp Balsam ApplicationDefinition.")

class RemoteForces(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/forces/forces.x"
        + " {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}"
        + " > out.txt 2>&1"
    )

    transfers = {
        "result": {
            "required": False,
            "direction": "out",
            "local_path": "forces.stat",
            "description": "Forces stat file",
            "recursive": False,
        }
    }

print("Defined RemoteForces Balsam ApplicationDefinition.")

LibensembleApp.sync()
RemoteForces.sync()

print("Synced each app with the Balsam service.")
