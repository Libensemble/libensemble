### Forces Mini-App

A system of charged particles is initialized and simulated over a number of time-steps.

Particles' position and charge are initiated using a random stream.
Particles are replicated on all ranks.
**Each rank** computes forces for a subset of particles (`O(N^2)` operations).
Particle force arrays are `allreduced` across ranks.
Particles are moved (replicated on each rank).
Total energy is appended to the forces.stat file.

Choose or modify a build line from `build_forces.sh` for the target platform and
run to build `forces.x`:

    ./build_forces.sh

To run forces as a standalone executable on `N` procs:

    mpirun -np N ./forces.x <NUM_PARTICLES> <NUM_TIMESTEPS> <SEED>

This is a good test to make sure forces is working, before running via libEnsemble.

The change the rate at which runs are declared bad runs (e.g.,~ for testing worker
kills), add a fourth argument between 0 and 1.
