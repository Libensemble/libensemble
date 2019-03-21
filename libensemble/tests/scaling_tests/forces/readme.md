## Running test run_libe_forces.py

Naive Electostatics Code Test

This is designed only as an artificial, highly conifurable test
code for a libEnsemble sim func.


### Forces Mini-App

A system of charged particles is set up and simulated over a number of time-steps.

Particles position and charge are initiated by a random stream.
Particles are replicated on all ranks. 
**Each rank** computes forces for a subset of particles (O(N^2))
Particle force arrays are allreduced across ranks.
Particles are moved (replicated on each rank)
Total energy is appended to file forces.stat

To run forces as a standalone executable on N procs:
    
    mpirun -np N ./forces.x <NUM_PARTICLES> <NUM_TIMESTEPS> <SEED>

    
### Running with libEnsemble.

A random sample of seeds is taken and used as imput to the sim func (forces miniapp).

Modify build_forces.sh for target platform and run to build forces.x

    ./build_forces.sh

To run with one manager and N-1 workers:

    mpirun -np N python run_libe_forces.py
    
Application parameters can be adjusted in the file run_libe_forces.py.

To remove output before the next run:

    ./cleanup.sh


### Using batch scripts

The scripts are set up assuming a conda environment. To use script directly you will need to replace the following templated values:

	<projectID> in the COBALT -A directive with your project ID.

	<conda_env_name> is the name of your conda environment.

and in Balsam scripts:

	<dbase_name> The name of an initialized balsam database.
	             (with max_connections enough for number of workers)

The included scripts are.

* theta_submit_mproc.sh:

	Example Theta submission script to run libEnsemble in central mode on the Theta launch (MOM) nodes with multiprocessing worker concurrency.

* summit_submit_mproc.sh:

	Example Summit submission script to run libEnsemble in central mode on the Summit launch nodes with multiprocessing worker concurrency.

* theta_submit_balsam.sh:

	Example Theta submission script to run libEnsemble in central mode with MPI worker concurrency using Balsam. In this case libEnsemble manager and workers run on compute nodes and sumit jobs via Balsam.

#### Plotting Options

If either of the plotting options in the submission scripts are set to true, the scripts must be in the directory specified by PLOT_DIR. These scripts can be found in the libEnsemble project in the postproc_scripts/ directory.

        export LIBE_PLOTS=true
        export BALSAM_PLOTS=true

#### Note on theta_submit_balsam.sh

Adjusting the node/core/worker count.: The NUM_WORKERS variable is only currently used if libEnsemble is running on one node, in which case it should be one less than the number of nodes in the job allocation (leaving one dedicated node to run libEnsemble). If more workers are used then the variables NUM_NODES and RANKS_PER_NODE need to be explicitly set (these are for libensemble which will require one task for the manager and the rest will be workers). The total node allocation (in the COBALT -n directive) will need to be the number of nodes for libEnsemble + number of nodes for each worker to launch jobs to.




