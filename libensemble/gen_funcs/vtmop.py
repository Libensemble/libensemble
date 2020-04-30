"""
Wrapper for the VTMOP genfuncs drivers, for solving multiobjective
optimization problems with arbitrary number of objectives.
"""
import numpy as np
from scipy.io import FortranFile  # for reading/writing unformatted binary data
from os import system  # for issuing batch commands


def vtmop_gen(H, persis_info, gen_specs, _):
    """
    This generator function solves multiobjective optimization problems
    with d design variables (subject to simple bound constraints) and
    p objectives using VTMOP.

    This requires that VTMOP be installed and in the system PATH.
    To do so, download VTMOP (contact thchang@vt.edu), and use the
    command

    $ make genfuncs

    to build the generator functions. Next, run the command

    $ export PATH=$PATH:`pwd`

    from the VTMOP source/build directory to add VTMOP to your system
    PATH.

    This generator alternates between generating large batches of size
    gen_specs['search_batch_size'] to explore design regions, and small
    batches of size gen_specs['opt_batch_size'] to fill in gaps on the
    Pareto front. An initial search size can also be specified by using
    gen_specs['first_batch_size'].

    gen_specs['ub'] and gen_specs['lb'] must specify upper and lower
    bound constraints on each design variable. The number of design variables
    is inferred from len(gen_specs['ub']). gen_specs['num_obj']
    specifies the number of objectives. The problem dimension is inferred
    based on the length of the gen_specs['lb'].

    Several unformatted binary files (vtmop.io, vtmop.dat, and vtmop.chkpt)
    will be generated in the calling directory to pass information between
    libEnsemble and VTMOP. 
    """
    # First get the problem dimensions and data
    ub = gen_specs['ub']  # upper bounds
    lb = gen_specs['lb']  # lower bounds
    d = len(lb)  # design dimension
    p = gen_specs['num_obj']  # objective dimension
    snb = gen_specs['search_batch_size']  # preferred batch size for searching
    onb = gen_specs['opt_batch_size'] # preferred batch size for optimization
    inb = gen_specs['first_batch_size'] # batch size for first iteration
    n = np.size(H['f'][:, 0])  # size of database in the history array

    if len(H) == 0:
        # Write initialization data to the vtmop.io file for VTMOP_INIT
        fp1 = FortranFile('vtmop.io', 'w')
        fp1.write_record(np.array([np.int32(d), np.int32(p), np.int32(inb)]))
        fp1.write_record(np.array([np.array(lb, dtype=np.float64),
                         np.array(ub, dtype=np.float64)]))
        fp1.close()
        system("vtmop_initializer")
    else:
        # Write unformatted problem dimensions to the vtmop.io file
        fp1 = FortranFile('vtmop.io', 'w')
        fp1.write_record(np.array([np.int32(d), np.int32(p), np.int32(n),
                         np.int32(snb), np.int32(onb)]))
        fp1.write_record(np.array([np.array(lb, dtype=np.float64),
                         np.array(ub, dtype=np.float64)]))
        fp1.close()
        # Write unformatted history to the vtmop.dat file, to be read by VTMOP
        fp2 = FortranFile('vtmop.dat', 'w')
        fp2.write_record(np.array([np.int32(d), np.int32(p)]))
        for i in range(n):
            toadd = np.zeros(d+p)
            toadd[:d] = np.float64(H['x'][i, :])
            toadd[d:] = np.float64(H['f'][i, :])
            fp2.write_record(toadd)
            ## Debug statements below
            #if (np.float64((H['f'][i,:]) == np.zeros(p)).all()):
            #    print('here')
        fp2.close()
        # Call VTMOP from command line
        system("vtmop_generator")

    # Read unformatted list of candidates from vtmop.io file
    fp1 = FortranFile('vtmop.io', 'r')
    cand_pts = fp1.read_record(np.float64)
    fp1.close()

    # Get the true batch size
    b = cand_pts.size // d

    # Read record
    O = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0, b):
        O['x'][i] = cand_pts[d*i:d*(i+1)]

    return O, persis_info
