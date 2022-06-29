#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt

# The following is not explicitly called but is needed for 3d plotting to work
# with older versions of python/matplotlib. It is not needed for python3.8 with
# matplotlib version 3.2.1.
from mpl_toolkits import mplot3d  # noqa


# Loop through objective points in f and extract the Pareto front.
# input: f is a list (dimensions n X p) of n p-dimensional objective points.
# output: a new list (dimensions m X p, m<n) of m nondominate points from f.
def get_pf(f):
    eps = 10.0 ** (-8.0)
    [n, p] = np.shape(f_vals)
    pareto_f = []
    # Loop over all entries in f
    for p in f_vals:
        dom = False
        # Check whether any q in f_vals dominates p
        for q in f_vals:
            # If they are equal, there is no dominance
            if np.linalg.norm(p - q) < eps:
                continue
            # If q dominates p + a small perturbation, there is dominance
            elif (q < p + eps).all():
                dom = True
        # Add nondominated points to the list
        if not dom:
            pareto_f.append(p)
    # Return the nondominated set
    return np.asarray(pareto_f)


# Load the data
if len(sys.argv) > 1:
    hist = np.load(sys.argv[1])
else:
    print("You need to supply an .npy file - aborting")
    sys.exit()

# Get first n results
if len(sys.argv) > 2:
    n = int(sys.argv[2])
    f_vals = hist["f"][:n]
    pts = get_pf(f_vals)
else:
    print("You need to supply the budget - aborting")
    sys.exit()

# Plot the contents of argv[1] when P=3.
obj1 = []
obj2 = []
obj3 = []
for i in range(pts[:, 0].size):
    obj1.append(pts[i, 0])
    obj2.append(pts[i, 1])
    obj3.append(pts[i, 2])

# Set up the figure environment.
fig = plt.figure()
ax = plt.axes(projection="3d")

# Scatter the Pareto points.
ax.scatter(obj1, obj2, obj3)
ax.set_xlabel("F1")
ax.set_ylabel("F2")
ax.set_zlabel("F3")

# Print the title.
plt.title("Tradeoff curve between objectives F1, F2, and F3")

# Display the figure.
plt.show()
