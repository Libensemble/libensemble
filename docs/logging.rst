Logging
=======

The libEnsemble logger uses the standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

The default level is DEBUG as this can provide useful diagnostics. However, INFO level includes information such as how jobs are launched and when jobs are killed. libEnsemble produces logging to the file ensemble.log. 

The logging level can be changed at the top of the calling scripts E.g::

    from libensemble import libE_logger
    libE_logger.set_level('INFO')
