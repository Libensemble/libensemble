#!/usr/bin/env python3
"""Stand-in for CGYRO, used to test the workflow without CGYRO.

Reads ./input.cgyro and writes out.cgyro.freq and out.cgyro.info with a synthetic gamma(delta, zeta, ky).
Environment variables select behavior:
  STUB_SLEEP              seconds to sleep before writing output (default 0)
  STUB_MAXTIME_KY         ky for which the first STUB_MAXTIME_ATTEMPTS attempts end at "max time"
  STUB_MAXTIME_ATTEMPTS   number of such attempts (default 0)
  STUB_FAIL_KY            ky for which the run exits with an error and writes no output
"""
import os
import re
import sys
import time

import numpy as np

with open("input.cgyro") as f:
    params = dict(re.findall(r"^\s*([A-Z_0-9]+)\s*=\s*(\S+)", f.read(), re.M))
ky, delta, zeta = (float(params[k]) for k in ("KY", "DELTA", "ZETA"))

attempts = int(open("stub_attempts").read()) + 1 if os.path.exists("stub_attempts") else 1
with open("stub_attempts", "w") as f:
    f.write(str(attempts))

time.sleep(float(os.environ.get("STUB_SLEEP", "0")))

if abs(ky - float(os.environ.get("STUB_FAIL_KY", "-1"))) < 1e-6:
    sys.exit(1)

gamma = ky * (0.5 + (delta + 0.2) ** 2 + 2 * (zeta - 0.1) ** 2) * (1 + 0.3 * np.sin(10 * ky))
omega = -0.5 * ky
with open("out.cgyro.freq", "w") as f:
    for scale in (0.8, 0.95, 1.0):
        f.write(f"{omega * scale:.6e} {gamma * scale:.6e}\n")

hit_max_time = (
    abs(ky - float(os.environ.get("STUB_MAXTIME_KY", "-1"))) < 1e-6
    and attempts <= int(os.environ.get("STUB_MAXTIME_ATTEMPTS", "0"))
)
with open("out.cgyro.info", "w") as f:
    f.write(" step  err\n 1 0.1\n\n")
    f.write("EXIT: (CGYRO) Linear terminated at max time\n" if hit_max_time else "EXIT: (CGYRO) Linear converged\n")
