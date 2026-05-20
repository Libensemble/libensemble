import logging
import threading

logger = logging.getLogger(__name__)


class GCSession:
    """Per-process singleton cache for Globus Compute executors.

    Caches executor instances keyed by endpoint_id, ensuring only one
    executor per endpoint per process. Thread-safe via ``threading.Lock``.
    """

    _instances: dict[str, tuple] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create_executor(cls, endpoint_id: str):
        """Get or create a cached executor for the given endpoint.

        Unlike :meth:`get_or_create`, this does **not** register a function.
        """
        with cls._lock:
            if endpoint_id in cls._instances:
                return cls._instances[endpoint_id][0]

            executor = cls._create_executor(endpoint_id)
            if executor is None:
                return None

            cls._instances[endpoint_id] = (executor, None)
            return executor

    @classmethod
    def get_or_create(cls, endpoint_id: str, func):
        """Get or create a cached ``(executor, func_id)`` pair.

        The first call for an endpoint creates the executor and registers
        the callable. Subsequent calls return the cached pair (the
        registered function is re-used).
        """
        with cls._lock:
            if endpoint_id in cls._instances:
                executor, existing_fid = cls._instances[endpoint_id]
                if existing_fid is not None:
                    return executor, existing_fid
                # Executor exists but no function registered yet
                func_id = executor.register_function(func)
                cls._instances[endpoint_id] = (executor, func_id)
                return executor, func_id

            executor = cls._create_executor(endpoint_id)
            if executor is None:
                return None, None

            func_id = executor.register_function(func)
            cls._instances[endpoint_id] = (executor, func_id)
            return executor, func_id

    @classmethod
    def register_function(cls, endpoint_id: str, func):
        """Register an additional function with an existing executor.

        Returns ``(executor, func_id)``. Unlike :meth:`get_or_create`,
        this always registers and never caches the func_id (caller should
        cache it themselves).
        """
        executor = cls.get_or_create_executor(endpoint_id)
        if executor is None:
            return None, None
        func_id = executor.register_function(func)
        return executor, func_id

    @classmethod
    def _create_executor(cls, endpoint_id: str):
        try:
            from globus_compute_sdk import Executor
        except ModuleNotFoundError:
            logger.warning("Globus Compute use detected but Globus Compute not importable. " "Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        return Executor(endpoint_id=endpoint_id)

    @classmethod
    def clear(cls):
        """Clear the cache (primarily for testing)."""
        with cls._lock:
            cls._instances.clear()
