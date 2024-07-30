If needed, run /global/cfs/cdirs/m4493/shudson/gx/libe/setup.sh to load modules

If using the venv gx_libe_env:
> source /global/cfs/cdirs/m4493/shudson/gx/gx_libe_env/bin/activate

Else, if using your own python environment install libensemble:

> cd libensemble
> pip install -e .

Enter run_libe_gx_inputs directory.

In run_gx.py make sure paths to cyclone.in and gx are correct,
where cyclone.in has templated values for input parameters.

To run with one node and four concurrent gx simulations on a node:

> sbatch cyclone-job.sh

The run line:
python run_gx.py -n 4

runs 4 workers in parallel. Each will run GX using one processor and one GPU.

The simulation workers will each use the number of GPUs available.

E.g., to run on one node with two workers running simulations, each using
2 MPI ranks and 2 GPUs:

python run_gx.py -n 2

