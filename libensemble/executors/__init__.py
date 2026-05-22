from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor

# FluxExecutor is optional - requires flux-core Python bindings
try:
    from libensemble.executors.flux_executor import FluxExecutor  # noqa: F401

    __all__ = ["Executor", "MPIExecutor", "FluxExecutor"]
except ImportError:
    # flux-core not available - FluxExecutor won't be importable
    __all__ = ["Executor", "MPIExecutor"]
