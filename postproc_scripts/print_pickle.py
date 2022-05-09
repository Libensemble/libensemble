#!/usr/bin/env python
import sys
import pickle
from pprint import pprint

pretty = True  # Default

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("You need to supply an .pickle file - aborting")
    sys.exit()

if len(sys.argv) > 2:
    if sys.argv[2] == "pretty" or "p":
        pretty = True
    if sys.argv[2] == "raw" or "r":
        pretty = False

with open(filename, "rb") as f:
    data = pickle.load(f)

if pretty:
    pprint(data, compact=True)
else:
    print(data)
