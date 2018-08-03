Work
====

Dictionary with integer keys ``i`` and dictionary values to be given to worker ``i``. 
``Work[i]`` has the following form::


    Work[i]: [dict]:

        Required keys :    
        'persis_info' [dict]: Any persistent info to be sent to worker 'i' 

        'H_fields' [list]: The field names of the history 'H' to be sent to worker 'i' 

        'tag' [int]: 'EVAL_SIM_TAG' (resp. 'EVAL_GEN_TAG') if worker 'i' is to call sim_func (resp. gen_func) 

        'libE_info' [dict]: This information is sent to and returned from the worker to help libEnsemble quickly update the 'H' and 'W'. 
            Available keys are:

            H_rows' [list of ints]: History rows to send to worker 'i'

            blocking' [list of ints]: Workers to be blocked by the calculation given to worker 'i'

            persistent' [bool]: True if worker 'i' will enter persistent mode 
        
        
:Examples:

:See Also:



