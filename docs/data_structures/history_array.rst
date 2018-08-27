.. _datastruct-history-array:

history array
=============

Stores the history of the output from gen_f, sim_f, and alloc_f:: 

    H: numpy structured array
        History array storing rows for each point. 

Fields in ``H`` include those specified in ``sim_specs['out']``,
``gen_specs['out']``, and ``alloc_specs['out']``. All values are initiated to
0 for integers, 0.0 for floats, and False for booleans.



Below are the protected fields used in ``H``

..  literalinclude:: ../../libensemble/libE_fields.py
    :lines: 4-
        
:Examples:

See example :doc:`sim_specs<./sim_specs>`, :doc:`gen_specs<./gen_specs>`, and :doc:`alloc_specs<./alloc_specs>`.
