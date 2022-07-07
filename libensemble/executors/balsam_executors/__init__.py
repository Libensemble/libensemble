import importlib

# try:
#     if importlib.util.find_spec("balsam"):  # Balsam 0.7.0 onward (Balsam 2)
#         from libensemble.executors.balsam_executors.balsam_executor import BalsamExecutor

# except (ModuleNotFoundError, ImportError):
#     try:
#         if importlib.util.find_spec("balsam-flow"):  # Balsam up through 0.5.0
#             from libensemble.executors.balsam_executors.legacy_balsam_executor import LegacyBalsamMPIExecutor
#     except (ModuleNotFoundError, ImportError):
#         pass

if importlib.util.find_spec("balsam"):

    try:
        from libensemble.executors.balsam_executors.balsam_executor import BalsamExecutor
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        from libensemble.executors.balsam_executors.legacy_balsam_executor import LegacyBalsamMPIExecutor
    except (ModuleNotFoundError, ImportError):
        pass

    __all__ = ["LegacyBalsamMPIExecutor", "BalsamExecutor"]
