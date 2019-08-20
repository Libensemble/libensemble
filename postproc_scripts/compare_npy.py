#!/usr/bin/env python

import sys
import numpy as np

if len(sys.argv) > 1:
    results = np.load(sys.argv[1])
else:
    print('You need to supply an .npy file - aborting')
    sys.exit()

exp_results_file = "../expected.npy"
exp_results = np.load(exp_results_file)
exclude_fields = ['gen_worker', 'sim_worker', 'gen_time', 'given_time']  # list of fields to ignore
locate_mismatch = True

compare_fields = tuple(filter(lambda x: x not in exclude_fields, exp_results.dtype.names))

match = all([np.allclose(exp_results[name], results[name], equal_nan=True) for name in compare_fields])
print('Compare results: {}\n'.format(match))

if not locate_mismatch:
    assert match, 'Error: Results do NOT match'

if not match:
    for name in compare_fields:
        for i in range(len(results)):
            assert np.isclose(exp_results[name][i], results[name][i], equal_nan=True), \
                'Mismatch in row ' + str(i) + ' field: ' + name + '. ' \
                + str(exp_results[name][i]) + ' ' + str(results[name][i])
