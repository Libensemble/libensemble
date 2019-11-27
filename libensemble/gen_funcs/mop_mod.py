"""
Wrapper for the MOP_MOD genfuncs drivers, for solving multiobjective
optimization problems with arbitrary number of objectives.
"""
import numpy as np
from scipy.io import FortranFile  # for reading/writing unformatted binary data
from os import system  # for issuing batch commands


def mop_mod_gen(H, persis_info, gen_specs, _):
    """
    This generator function solves multiobjective optimization problems
    with d design variables (subject to simple bound constraints) and
    p objectives using MOP_MOD.

    This requires that MOP_MOD be installed and in the system PATH.
    To do so, download MOP_MOD (contact thchang@vt.edu), and use the
    command

    $ make genfuncs

    to build the generator functions. Next, run the command

    $ export PATH=$PATH:`pwd`

    from the MOP_MOD source/build directory to add MOP_MOD to your system
    PATH.

    This generator alternates between generating large batches of size
    gen_specs['gen_batch_size'] to explore design regions, and small
    batches of undetermined size to fill in gaps on the Pareto front.

    gen_specs['ub'] and gen_specs['lb'] must specify upper and lower
    bound constraints on each design variable. The number of design variables
    is inferred from len(gen_specs['ub']). gen_specs['num_obj']
    specifies the number of objectives.

    Several unformatted binary files (mop.io, mop.dat, and mop.chkpt) will
    be generated in the calling directory to pass information between
    libEnsemble and MOP_MOD. 
    """
    # First get the problem dimensions and data
    ub = gen_specs['ub']  # upper bounds
    lb = gen_specs['lb']  # lower bounds
    d = len(lb)  # design dimension
    p = gen_specs['num_obj']  # objective dimension
    nb = gen_specs['gen_batch_size']  # preferred batch size
    inb = gen_specs['first_batch_size'] # batch size for first iteration
    n = np.size(H['f'][:, 0])  # size of database in the history array

    if len(H) == 0:
        # Write initialization data to the mop.io file for MOP_INIT
        fp1 = FortranFile('mop.io', 'w')
        fp1.write_record(np.int32(d), np.int32(p), np.int32(inb))
        fp1.write_record(np.array(lb, dtype=np.float64),
                         np.array(ub, dtype=np.float64))
        fp1.close()
        system("mop_initializer")
    else:
        # Write unformatted problem dimensions to the mop.io file
        fp1 = FortranFile('mop.io', 'w')
        fp1.write_record(np.int32(d), np.int32(p), np.int32(n), np.int32(nb))
        fp1.write_record(np.array(lb, dtype=np.float64),
                         np.array(ub, dtype=np.float64))
        fp1.close()
        # Write unformatted history to the mop.dat file, to be read by MOP_MOD
        fp2 = FortranFile('mop.dat', 'w')
        fp2.write_record(np.int32(d), np.int32(p))
        for i in range(n):
            fp2.write_record(np.float64(H['x'][i, :]), np.float64(H['f'][i, :]))
        fp2.close()
        # Call MOP_MOD from command line
        system("mop_generator")

    # Read unformatted list of candidates from mop.io file
    fp1 = FortranFile('mop.io', 'r')
    cand_pts = fp1.read_record(np.float64)
    fp1.close()

    # Get the true batch size
    b = cand_pts.size // d

    # Read record
    O = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0, b):
        O['x'][i] = cand_pts[d*i:d*(i+1)]

    return O, persis_info
