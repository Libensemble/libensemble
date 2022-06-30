#!/usr/bin/env python

import sys
import numpy as np

if len(sys.argv) > 1:
    results = np.load(sys.argv[1])
else:
    print("You need to supply an .npy file - aborting")
    sys.exit()

done_only = False
dtype_only = False

if len(sys.argv) > 2:
    if sys.argv[2] == "done":
        done_only = True
    elif sys.argv[2] == "dtype":
        dtype_only = True
else:
    print(repr(results))

if done_only:
    count = 0
    for line in results:
        if line["sim_ended"]:
            count += 1

    results_filtered = np.zeros(count, dtype=results.dtype)
    count = 0
    for i, line in enumerate(results):
        if line["sim_ended"]:
            results_filtered[count] = results[i]
            count += 1

    print(repr(results_filtered))

elif dtype_only:
    print(results.dtype.names)
    print(type(results.dtype.names))
