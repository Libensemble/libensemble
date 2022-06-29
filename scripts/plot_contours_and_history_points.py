"""
This function plots
- Rosenbrock function contours on lb = [-3, 2] ub = [3, 2]
- Batches of requested points in the sys.argv[1] filename

batch size must be declared
num_batches is determined by the unique "gen_ended_time" values

Working case:
- copy this file to the regression test directory
- Run:
      mpiexec -np 4 python3 test_persistent_uniform_sampling.py
- Run
      python plot_contours_and_history_points.py XXX.npy
  where XXX.npy is the last history file saved by the regression test
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

from libensemble.sim_funcs.rosenbrock import EvaluateFunction as obj_f

delta = 0.025
x = np.arange(-3.0, 3.0 + delta, delta)
y = np.arange(-2.0, 2.0 + delta, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(np.shape(X)[0]):
    for j in range(np.shape(X)[1]):
        Z[i, j] = obj_f(np.array([X[i, j], Y[i, j]]), np.nan)

fig, ax = plt.subplots()
levels = [0.1, 1, 5, 10, 30, 50, 70]
colormap = plt.get_cmap("hsv")
scale = 100  # Scaling Rosenbrock function values makes it much easier to see meaningful contours.


CS = ax.contourf(X, Y, Z / 100, levels, cmap=colormap, alpha=0.2)
CS = ax.contour(X, Y, Z / 100, levels, cmap=colormap)
ax.clabel(CS, inline=True, fontsize=10)

hist_filename = sys.argv[1]
H = np.load(hist_filename)

num_batches = len(np.unique(H["gen_ended_time"]))
pts = H["x"]
batch = 10

for i in range(num_batches):
    ax.scatter(
        pts[i * batch : (i + 1) * batch, 0],
        pts[i * batch : (i + 1) * batch, 1],
        color="b",
        edgecolors="k",
    )
    plt.savefig("plot_after_batch_" + str(i))
