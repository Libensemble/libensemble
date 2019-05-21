
.. _datastruct-alloc-specs:

alloc_specs
===========

Allocation function specifications to be set in user calling script and passed to libE.libE()::

    alloc_specs: [dict, optional] :

        Required keys :  
        
        'alloc_f' [func] :
            Default: give_sim_work_first
            
        Optional keys :
        
        'out' [list of tuples] :
            Default: [('allocated',bool)]


:Notes:

* The alloc_specs has the default keys as given above, but may be overidden by the user.
* The tuples defined in the 'out' list are entered into the master :ref:`history array<datastruct-history-array>`


:Examples:
