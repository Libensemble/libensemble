#!/usr/bin/env python

"""Script to compare libEnsemble history arrays in files.

E.g., ./compare_npy.py out1.npy out2.npy

If two *.npy files are provided they are compared with each other.
If one *.npy file is provided if is compared with a hard-coded expected file
(by default located at ../expected.npy)

Default tolerances used for comparison are (rtol=1e-05, atol=1e-08). These
can be overwritten with -r (--rtol) and -a (--atol) flags.

E.g., ./compare_npy.py out1.npy out2.npy -r 1e-03

Nans compare as equal. Variable fields (such as those containing a time)
are ignored. In some cases you may have to ignore further user-defined fields

"""
import sys
import numpy as np
import argparse

desc = "Script to compare libEnsemble history arrays in files"
example = """examples:

 ./compare_npy.py out1.npy out2.npy
 ./compare_npy.py out1.npy out2.npy --rtol 1e-03 --atol 1e-06
 """

exclude_fields = ["gen_worker", "sim_worker", "gen_ended_time", "sim_started_time"]  # list of fields to ignore
locate_mismatch = True

parser = argparse.ArgumentParser(description=desc, epilog=example, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-r", "--rtol", dest="rtol", type=float, default=1e-05, help="rel. tolerance")
parser.add_argument("-a", "--atol", dest="atol", type=float, default=1e-08, help="abs. tolerance")
parser.add_argument("args", nargs="*", help="*.npy files to compare")
args = parser.parse_args()

rtol = args.rtol
atol = args.atol
files = args.args

if len(files) >= 1:
    results = np.load(files[0])
    exp_results = np.load(files[1]) if len(files) >= 2 else np.load("../expected.npy")
else:
    parser.print_help()
    sys.exit()

compare_fields = tuple(filter(lambda x: x not in exclude_fields, exp_results.dtype.names))
match = all(
    [np.allclose(exp_results[name], results[name], rtol=rtol, atol=atol, equal_nan=True) for name in compare_fields]
)

print(f"Compare results: {match}\n")

if not locate_mismatch:
    assert match, "Error: Results do NOT match"

if not match:
    for name in compare_fields:
        for i in range(len(results)):
            assert np.allclose(exp_results[name][i], results[name][i], rtol=rtol, atol=atol, equal_nan=True), (
                "Mismatch in row "
                + str(i)
                + " field: "
                + name
                + ". "
                + str(exp_results[name][i])
                + " "
                + str(results[name][i])
            )
