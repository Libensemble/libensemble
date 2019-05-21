Logging
=======

The libEnsemble logger uses the standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

The default level is INFO, which includes information such as how jobs are launched and when jobs are killed. To gain additional diagnostics, logging level can be set to DEBUG. libEnsemble produces logging to the file ensemble.log by default. A log file name can also be supplied.

E.g. To change The logging level to DEBUG, provide the following in your the calling scripts::

    from libensemble import libE_logger
    libE_logger.set_level('DEBUG')

Logging API
-----------

.. automodule:: libE_logger
  :members:
  :no-undoc-members:
