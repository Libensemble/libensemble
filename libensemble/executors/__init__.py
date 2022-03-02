from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor

import os
import pkg_resources

try:
    if pkg_resources.get_distribution('balsam-flow'):
        if 'BALSAM_DB_PATH' in os.environ:
            from libensemble.executors.balsam_executor import BalsamMPIExecutor
        else:
            from libensemble.executors.new_balsam_executor import NewBalsamExecutor

except ModuleNotFoundError:  # One version of Balsam installed, but not the other
    pass

__all__ = ['BalsamMPIExecutor', 'Executor', 'MPIExecutor', 'NewBalsamExecutor']
