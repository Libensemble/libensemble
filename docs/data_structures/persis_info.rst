.. _datastruct-persis-info:

persis_info
===========

Supply persistent information to libEnsemble::

    persis_info: [dict] :
        Dictionary containing persistent info

Holds data that is passed to and from workers updating some state information. A typical example
is a randon number generator to be used in consecutive calls to a generator.

If worker ``i`` sends back ``persis_info``, it is stored in ``persis_info[i]``. This functionality
can be used to, for example, pass a random stream back to the manager to be included in future work
from the allocation function. 

:Examples:

From: libEnsemble/tests/regression_tests/test_6-hump_camel_aposmm_LD_MAA.py::

    persis_info = {'next_to_give':0} # used in alloc_funcs/fast_alloc_to_aposmm.py to store the next entry in H to give
    persis_info['total_gen_calls'] = 0 # used in alloc_funcs/fast_alloc_to_aposmm.py to count total gen calls

    for i in range(MPI.COMM_WORLD.Get_size()):
        persis_info[i] = {'rand_stream': np.random.RandomState(i)} # used as a random number stream for each worker

