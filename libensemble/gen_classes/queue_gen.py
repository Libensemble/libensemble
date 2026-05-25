"""Queue-backed generator: work comes from an external producer (e.g. an MCP
tool, an LLM agent, a REST API) via an input queue. Completed results go back
out via an output queue.

This lets libE run in 'service mode' — driven by an external loop instead of
its own gen function.

Usage:
    from queue import Queue
    from gest_api.vocs import VOCS
    from libensemble.gen_classes.queue_gen import QueueGenerator

    in_q = Queue()
    out_q = Queue()
    gen = QueueGenerator(VOCS(variables={"x": [-1, 1]}),
                         input_queue=in_q, output_queue=out_q)
    in_q.put({"x": 0.5})  # external producer feeds work
"""
import threading
import time
from queue import Empty, Queue
from typing import Any, List, Optional

from gest_api import Generator
from gest_api.vocs import VOCS


_SHUTDOWN = object()  # sentinel: external producer signals "no more work"


class QueueGenerator(Generator):
    """Generator that pulls work from an external Queue and pushes results back.

    suggest(n):
      - Blocks up to ``poll_timeout`` waiting for the FIRST item (so the libE
        manager doesn't spin hot when nothing is queued).
      - Then drains up to ``n - 1`` more items non-blockingly.
      - Returns [] on timeout (libE will call again).
      - Returns [] if a shutdown sentinel is seen.

    ingest(results):
      - Forwards each result dict to the output queue verbatim.
    """

    def __init__(
        self,
        vocs: VOCS,
        *,
        input_queue: Queue,
        output_queue: Queue,
        poll_timeout: float = 1.0,
    ):
        self.vocs = vocs
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.poll_timeout = poll_timeout
        self._shutdown_seen = False
        super().__init__(vocs)

    def _validate_vocs(self, vocs: VOCS) -> None:
        assert len(self.vocs.variable_names), "VOCS must contain variables."

    def suggest(self, num_points: Optional[int]) -> List[dict]:
        if self._shutdown_seen:
            return []
        n = num_points or 1
        items: List[dict] = []
        try:
            first = self.input_queue.get(timeout=self.poll_timeout)
        except Empty:
            return []
        if first is _SHUTDOWN:
            self._shutdown_seen = True
            return []
        items.append(first)
        for _ in range(n - 1):
            try:
                nxt = self.input_queue.get_nowait()
            except Empty:
                break
            if nxt is _SHUTDOWN:
                self._shutdown_seen = True
                break
            items.append(nxt)
        return items

    def ingest(self, results: List[dict]) -> None:
        for r in results:
            self.output_queue.put(r)

    def finalize(self, results: List[dict] = None, *args: Any, **kwargs: Any):
        if results:
            self.ingest(results)
        return None

    @staticmethod
    def shutdown_sentinel():
        """External producer puts this on the input queue to signal end."""
        return _SHUTDOWN


class QueueService:
    """Service-mode wrapper: spawns libE in a thread with a QueueGenerator and
    exposes submit/get_completed/shutdown to an external producer.

    Hides the queue/thread/generator plumbing every producer would otherwise
    repeat. The producer just creates a service and feeds it work:

        service = QueueService(vocs, sim_specs, libE_specs, exit_criteria)
        service.start()
        service.submit({"x": 1.0})
        for r in service.get_completed():
            ...
        service.shutdown()
    """

    def __init__(
        self,
        vocs: VOCS,
        sim_specs,
        libE_specs,
        exit_criteria,
        *,
        persis_in: Optional[List[str]] = None,
        batch_size: int = 0,
        poll_timeout: float = 1.0,
    ):
        from libensemble import Ensemble
        from libensemble.specs import GenSpecs

        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self._gen = QueueGenerator(
            vocs,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            poll_timeout=poll_timeout,
        )
        gen_specs = GenSpecs(
            generator=self._gen,
            vocs=vocs,
            persis_in=persis_in or [],
            batch_size=batch_size,
        )
        self._ensemble = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Spawn the libE thread."""
        self._thread = threading.Thread(target=self._ensemble.run, daemon=True)
        self._thread.start()

    def submit(self, item: dict) -> None:
        """Submit one work item."""
        self.input_queue.put(item)

    def get_completed(self) -> List[dict]:
        """Drain all completed results (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self.output_queue.get_nowait())
            except Empty:
                break
        return out

    def collect_results(self, n: int, timeout: float = 60.0) -> List[dict]:
        """Block-drain until ``n`` results collected or ``timeout`` elapses.
        Returns whatever was collected (may be < n on timeout)."""
        results: List[dict] = []
        deadline = time.time() + timeout
        while len(results) < n and time.time() < deadline:
            try:
                results.append(self.output_queue.get(timeout=1))
            except Empty:
                pass
        return results

    def stream_results(self, n: Optional[int] = None, timeout: float = 60.0):
        """Yield results as they arrive. Stops after ``n`` yielded or
        ``timeout`` seconds elapse with no new result. ``n=None`` streams
        indefinitely until timeout."""
        deadline = time.time() + timeout
        yielded = 0
        while (n is None or yielded < n) and time.time() < deadline:
            try:
                r = self.output_queue.get(timeout=1)
            except Empty:
                continue
            deadline = time.time() + timeout  # reset on activity
            yielded += 1
            yield r

    def shutdown(self, wait: bool = False) -> None:
        """Signal libE to stop accepting new work and drain in-flight.
        If ``wait=True``, block until the libE thread exits."""
        self.input_queue.put(_SHUTDOWN)
        if wait:
            self.join()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for libE thread to exit."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
