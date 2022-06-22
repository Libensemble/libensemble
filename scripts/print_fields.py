#!/usr/bin/env python

import sys
import numpy as np
import argparse

desc = "Script to print selected fields of a libEnsemble history array from a file"
example = """examples:
 ./print_fields.py out1.npy --fields sim_id x f returned

 If no fields are supplied the whole array is printed.

 You can filter by using conditions, but only boolean are supported currently e.g:
 ./print_fields.py out1.npy -f sim_id x f -c given ~returned

 would show lines where given is True and returned is False.
 """

np.set_printoptions(linewidth=1)

parser = argparse.ArgumentParser(description=desc, epilog=example, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-f", "--fields", nargs="+", default=[])
parser.add_argument("-c", "--condition", nargs="+", default=[])
parser.add_argument("-s", "--show-fields", dest="show_fields", action="store_true")
parser.add_argument("args", nargs="*", help="*.npy file")
args = parser.parse_args()

fields = args.fields
npfile = args.args
cond = args.condition
show_fields = args.show_fields

if len(npfile) >= 1:
    np_array = np.load(npfile[0])
else:
    parser.print_help()
    sys.exit()

if not fields:
    fields = list(np_array.dtype.names)

if show_fields:
    print("Showing fields:", fields)

if cond:
    fltr = None
    for cn in cond:
        if cn[0] == "~":
            fltr = ~np_array[cn[1:]] if fltr is None else ~np_array[cn[1:]] & fltr
        else:
            fltr = np_array[cn] if fltr is None else np_array[cn] & fltr
    print(np_array[fields][fltr])
else:
    print(np_array[fields])
