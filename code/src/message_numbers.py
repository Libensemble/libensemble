STOP_TAG = 0                        # manager tells a worker to stop receiving work and exit
EVAL_SIM_TAG = 1                    # manager tells a worker to do a sim eval
EVAL_GEN_TAG = 2                    # manager tells a worker to do a gen eval
PERSIS_STOP = 4                     # manager tells a persistent worker to stop (and go back into general worker mode)
FINISHED_PERSISTENT_SIM_TAG = 11    # tells manager sim_f is done with persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12    # tells manager gen_f is done with persistent mode
