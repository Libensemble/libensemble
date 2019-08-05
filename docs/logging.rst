Logging
=======

The libEnsemble logger uses the standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

The default level is INFO, which includes information such as how jobs are launched
and when jobs are killed. To gain additional diagnostics, logging level can be set
to DEBUG. libEnsemble produces logging to the file ensemble.log by default. A log
file name can also be supplied.

E.g. To change the logging level to DEBUG, provide the following in your the calling scripts::

    from libensemble import libE_logger
    libE_logger.set_level('DEBUG')

Logger messages of WARNING level or higher are also displayed through stderr by default. This boundary can be adjusted as follows::

    from libensemble import libE_logger

    # Only display messages with level >= ERROR
    libE_logger.set_stderr_level('ERROR')

This feature can be effectively disabled by setting the stderr level to CRITICAL.


Logging API
-----------

.. automodule:: libE_logger
  :members:
  :no-undoc-members:
