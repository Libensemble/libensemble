import importlib.util
import warnings

from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor

if importlib.util.find_spec("balsam"):
    try:
        from libensemble.executors.balsam_executor import BalsamExecutor  # noqa: F401

        __all__ = ["Executor", "MPIExecutor", "BalsamExecutor"]
        warnings.warn(
            "BalsamExecutor is deprecated, to be removed in v2.0. "
            + "For cross-site ensembles, use Globus Compute. See the docs for more information.",
            FutureWarning,
        )

    except (ModuleNotFoundError, ImportError, AttributeError):
        __all__ = ["Executor", "MPIExecutor"]
