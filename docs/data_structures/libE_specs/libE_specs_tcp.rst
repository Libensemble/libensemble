TCP
===

`Introduction <libE_specs.html>`__ \|\| `General <libE_specs_general.html>`__ \|\| `Directories <libE_specs_directories.html>`__ \|\| `Profiling <libE_specs_profiling.html>`__ \|\| **TCP** \|\| `History <libE_specs_history.html>`__ \|\| `Resources <libE_specs_resources.html>`__

**workers** [list]:
    TCP Only: A list of worker hostnames.

**ip** [str]:
    TCP Only: IP address for Manager's system.

**port** [int]:
    TCP Only: Port number for Manager's system.

**authkey** [str]:
    TCP Only: Authkey for Manager's system.

**workerID** [int]:
    TCP Only: Worker ID number assigned to the new process.

**worker_cmd** [list]:
    TCP Only: Split string corresponding to worker/client Python process invocation. Contains
    a local Python path, calling script, and manager/server format-fields for ``manager_ip``,
    ``manager_port``, ``authkey``, and ``workerID``. ``nworkers`` is specified normally.
