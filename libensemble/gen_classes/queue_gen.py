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
