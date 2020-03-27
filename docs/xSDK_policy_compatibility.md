# xSDK Community Policy Compatibility for libEnsemble

This document summarizes the efforts of libEnsemble
to achieve compatibility with the xSDK community policies.

**Website:** https://github.com/Libensemble/libensemble

### Mandatory Policies

[General libEnsemble Note](#liben-note)

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support xSDK community GNU Autoconf or CMake options. |N/A| libEnsemble is a Python package and provides a `setup.py` file for installation. This is compatible with Python's built-in installation feature (`python setup.py install`) and with the ubiquitous `pip` installer. libEnsemble is also in the Spack repository and can be installed with `spack install py-libensemble`. GNU Autoconf or CMake are unsuitable for a Python package.
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| libEnsemble has a test suite that includes both unit tests and regression tests that are run on every push to GitHub via [Travis CI](https://travis-ci.org/Libensemble/libensemble). In addition to this test suite, further scaling tests are manually run on HPC platforms including Cori, Theta, and Summit.
|**M3.** Employ user-provided MPI communicator (no MPI_COMM_WORLD). |Full|libEnsemble takes an MPI communicator as an option; if libEnsemble is configured for MPI mode, this provided communicator will be employed. If no communicator is given, a duplicate of MPI_COMM_WORLD is taken as a default. |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| libEnsemble is tested regularly, including prior to every release, on ALCF (Theta), OLCF (Summit) and NERSC (Cori) platforms. [M4 details](#m4-details)|
|**M5.** Provide a documented, reliable way to contact the development team. |Full| The libEnsemble team can be contacted through: 1) The public [issues page on GitHub](https://github.com/Libensemble/libensemble/issues). 2) [Slack](https://libensemble.slack.com). 3) The public email list libensemble@mcs.anl.gov. |
|**M6.** Respect system resources and settings made by other previously called packages (e.g., signal handling). |Full| libEnsemble does not modify system resources or settings. |
|**M7.** Come with an open source (BSD style) license. |Full| libEnsemble uses a 3-clause BSD license stated in the `LICENSE` file in the top level of the GitHub repository. |
|**M8.** Provide a runtime API to return the current version number of the software. |Full| The version can be returned within Python via: `libensemble.__version__`|
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| All libEnsemble symbols (e.g., functions, variables, modules, packages) begin with the prefix `libensemble.`. This prevents any namespace conflicts.|
|**M10.** Provide an xSDK team accessible repository (not necessarily publicly available). |Full| The libEnsemble repository is public and can be found at https://github.com/Libensemble/libensemble. Gitflow is used, along with pull requests, whereby only those with administrator privileges can accept pull requests into the master or develop branches. The workflow guidelines are provided in a `CONTRIBUTING.rst` file at the top level of the repository and a release process is given in the documentation. |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Full| All output from the libEnsemble core package, except for the raising of exceptions, is routed through a libEnsemble logger, which is isolated from the Python root logger. Log messages of type `MANAGER_WARNING` or above are duplicated to standard error by default to ensure they are not missed. This can be turned off through the API. The API also allows the user to change the logging verbosity level and the name of the log file. This would allow a user, for example, to append logging to an existing log file, or to keep it separate. libEnsemble contains no interactive input. libEnsemble creates the files `ensemble.log` and `libE_stats.txt`, but the creation of these files can be preempted. [M11 details](#m11-details)|
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| libEnsemble does not contain any other package's source code within. Note that Python packages are imported using the conventional `sys.path` system. Alternative instances of a package can be used by, for example, including in the `PYTHONPATH` environment variable.|
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full| The standard Python installation is used for Python dependencies. This installs external Python packages under `<install-prefix>/lib/python<X.Y>/site-packages/` When installed through Spack, the `<install-prefix>` is specific to each Python package. This is added to `PYTHONPATH` when the Spack module for that library is loaded.|
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full| There is no explicit use of pointers in libEnsemble, as Python handles pointers internally and depends on the install of Python (e.g., CPython), which will generally be 64-bit on supported systems. |
|**M15.** All xSDK compatibility changes should be sustainable. |Full| The xSDK-compatible package is in the standard release path. All the changes here should be sustainable. |
|**M16.** The package must support production-quality installation compatible with the xSDK install tool and xSDK metapackage. |Full|libEnsemble configure and install has full support from Spack. |

M4 details <a id="m4-details"></a>: libEnsemble is a Python code and so does
not directly use compilers. It does, however, use NumPy, SciPy and mpi4py which
use compiled extensions. The current CI tests of libEnsemble use the standard
CPython compatible builds of these extensions (which are built using the GNU
compilers). libEnsemble is also regularly tested using the Intel distribution
for Python.

libEnsemble is supported on Linux platforms and macOS. Windows platforms are
currently not supported.

M11 details <a id="m11-details"></a>: Note: The sub-packages in the libensemble
directory structure such as `sim_specs` and `gen_specs` may contain print
statements. These are considered examples for users, rather than core
libEnsemble packages.

A special exception exists in the `node_resources.py` module; part of
libEnsemble's resource detection infrastructure. The routine
`_print_local_cpu_resources()` can be launched by libEnsemble to probe
resources on a target node, and the output of this independent program is
captured by libEnsemble.

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Full| Yes (see M10 above). |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |Full| It is possible to run the test suite under Valgrind. While libEnsemble is Python code, this may be useful for compiled extensions that are imported. PYTHONMALLOC=malloc must be set on the run line. CPython also provides a suppression file.|
|**R3.** Adopt and document consistent system for error conditions/exceptions. |Full| libEnsemble defines and raises exceptions according to module. All exceptions on workers are passed to the manager for processing.  Warnings are handled by the logger. [R3 details](#r3-details)||
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Full| Python has built-in garbage collection that frees memory when it becomes unreferenced. When opening files, wherever possible, `with` expressions or `try/finally` blocks are used to ensure file handles are closed, even in the case of an error.|
|**R5.** Provide a mechanism to export ordered list of library dependencies. |Full| The dependencies for libEnsemble are given in `setup.py` and when pip install or pip setup.py egg_info are run, a file is created `libensemble.egg-info/requires.txt` containing the list of required and optional dependencies. If installing through pip, these will automatically be installed if they do not exist (`pip install libensemble` installs req. dependencies, while `pip install libensemble[extras]` installs both required and optional dependencies.|
|**R6.** Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Full| Dependencies are given in the documentation. In some cases, this includes a lower bound on the version number. These dependencies are also specified in the Spack package, and automatically resolved during installation.|
|**R7.** Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Full| These files are present in the top directory.|

R3 details <a id="r3-details"></a>: libEnsemble catches all exceptions
(explicitly raised and unexpected) from the manager and worker processes at the
libEnsemble level, resulting in libEnsemble dumping the key ensemble state to
files. In `mpi4py` mode, the default is to then call MPI_ABORT to prevent a
hang. However, this can be turned off (via the `libE_specs` argument). In the
case it is turned off, or if other communication modes are used, the exception
is then raised. The user can in turn catch these exceptions from their calling
script.

libEnsemble Note <a id="liben-note"></a>: The nature of libEnsemble's
interoperability with other libraries is different from typical xSDK libraries.
libEnsemble is a Python code and interaction with other libraries may take
several forms. These include: libEnsemble calling other libraries through
Python bindings, libEnsemble launching applications (possibly providing a
sub-communicator), libEnsemble being called from a Python level infrastructure,
libEnsemble being launched as part of a campaign level workflow, or libEnsemble
potentially being activated via a system call or embedded interpreter; a more
unconventional approach. This is, therefore, a good opportunity to consider
interoperability from a Python and broader workflow perspective.
