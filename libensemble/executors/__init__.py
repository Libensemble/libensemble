from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor

import pkg_resources

try:
    if pkg_resources.get_distribution("balsam"):  # Balsam 0.7.0 onward (Balsam 2)
        from libensemble.executors.balsam_executor import BalsamExecutor

except (ModuleNotFoundError, ImportError, pkg_resources.DistributionNotFound):
    try:
        if pkg_resources.get_distribution("balsam-flow"):  # Balsam up through 0.5.0
            from libensemble.executors.legacy_balsam_executor import LegacyBalsamMPIExecutor
    except (ModuleNotFoundError, ImportError, pkg_resources.DistributionNotFound):
        pass


__all__ = ["LegacyBalsamMPIExecutor", "Executor", "MPIExecutor", "BalsamExecutor"]
