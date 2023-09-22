## Tutorial

This example is referred to in the tutorial **Executor - Assign GPUs**.

When the generator creates parameters for each simulation, it sets a number
of GPUs required for the simulation. Resources are dynamically assigned to
the simulation workers.

https://libensemble.readthedocs.io/en/develop/tutorials/forces_gpu_tutorial.html

## QuickStart

Go to `forces_app` directory:

    cd ../forces_app

Compile **forces.x** using one of the GPU build lines in `build_forces.sh` or similar
for your platform (these will include -DGPU)

Then return here and run:

    python run_libe_forces.py --comms local --nworkers 5

This will run libEnsemble with five workers; one for the persistent generator, and
four for forces simulations (so four GPUs are required).

## Detailed instructions

See ../forces_gpu.
