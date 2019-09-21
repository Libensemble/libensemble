"""
Wrapper for MOP-MOD
"""
import numpy as np
from scipy.io import FortranFile # for reading/writing unformatted binary data
from os import system # for issuing batch commands


def mop_mod_wrapper(H, persis_info, gen_specs, _):
    """
    Generates ``gen_specs['gen_batch_size']`` points uniformly over the domain
    defined by ``gen_specs['ub']`` and ``gen_specs['lb']``.

    :See:
        ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``
    """
    # First get the problem dimensions and data
    ub = gen_specs['ub'] # upper bounds
    lb = gen_specs['lb'] # lower bounds
    d = len(lb) # design dimension
    p = gen_specs['num_obj'] # objective dimension
    nb = gen_specs['gen_batch_size'] # preferred batch size
    n = np.size(H['f'][:,0]) # size of database in the history array

    if len(H) == 0:
        # Write initialization data to the mop.dat file for MOP_INIT
        fp1 = FortranFile('mop.dat','w')
        fp1.write_record(np.int32(d), np.int32(p), np.int32(nb))
        fp1.write_record(np.array(lb, dtype=np.float64))
        fp1.write_record(np.array(ub, dtype=np.float64))
        fp1.close()
        system("./mop_init")
    else:
        # Write unformatted history to the mop.dat file, to be read by MOP_MOD
        fp1 = FortranFile('mop.dat','w')
        fp1.write_record(np.int32(d))
        fp1.write_record(np.int32(p))
        fp1.write_record(np.int32(n))
        fp1.write_record(np.int32(nb))
        for i in range(n):
            fp1.write_record(np.float64(H['x'][i,:]), np.float64(H['f'][i,:]))
        fp1.close()
        # Call MOP_MOD from command line
        system("./mop_gen")
    
    # Read unformatted list of candidates
    fp2 = FortranFile('mop.out','r')
    [nb,] = fp2.read_record(dtype=np.int32) # actual batch size may differ
    cand_pts = fp2.read_record(dtype=np.float64) # read in candidates
    fp2.close()

    O = np.zeros(nb, dtype=gen_specs['out'])
    for i in range(0,nb):
        O['x'][i,:] = cand_pts[d*i:d*(i+1)]
    #O['x'] = persis_info['rand_stream'].uniform(lb, ub, (nb, d))

    return O, persis_info
