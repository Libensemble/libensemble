#!/usr/bin/env python
import secrets

import numpy as np

from libensemble.ensemble import Ensemble

if __name__ == "__main__":
    forces = Ensemble()
    forces.from_yaml("funcx_forces.yaml")

    forces.sim_specs.user["remote_ensemble_dir"] += secrets.token_hex(nbytes=3)

    forces.gen_specs.user.update(
        {
            "lb": np.array([0]),
            "ub": np.array([32767]),
        }
    )

    forces.add_random_streams()

    forces.run()
