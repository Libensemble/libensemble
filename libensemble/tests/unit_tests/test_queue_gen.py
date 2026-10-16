import time
from queue import Queue
from unittest.mock import MagicMock

import pytest
from gest_api.vocs import VOCS

from libensemble.gen_classes.queue_gen import QueueGenerator, QueueService


def _make_gen():
    vocs = VOCS(variables={"x": [-1, 1]})
    return QueueGenerator(vocs, input_queue=Queue(), output_queue=Queue())


def test_getstate_strips_queues():
    # Simulates what happens when gen_specs (including the generator) is
    # pickled to spawn a sim-only worker under comms="local".
    gen = _make_gen()
    state = gen.__getstate__()
    assert state["input_queue"] is None
    assert state["output_queue"] is None
    # The live object itself is untouched.
    assert gen.input_queue is not None
    assert gen.output_queue is not None


def test_suggest_raises_clear_error_without_queues():
    # Simulates a QueueGenerator that ended up running somewhere other than
    # the manager (e.g. gen_on_worker=True, bypassing QueueService's guard)
    # and therefore only has the stripped-down pickled state.
    gen = _make_gen()
    gen.input_queue = None
    gen.output_queue = None
    with pytest.raises(RuntimeError, match="must run on the manager"):
        gen.suggest(1)


def test_ingest_raises_clear_error_without_queues():
    gen = _make_gen()
    gen.input_queue = None
    gen.output_queue = None
    with pytest.raises(RuntimeError, match="must run on the manager"):
        gen.ingest([{"x": 0.0}])


def test_queue_service_rejects_gen_on_worker():
    from libensemble.specs import LibeSpecs

    vocs = VOCS(variables={"x": [-1, 1]})
    libE_specs = LibeSpecs(nworkers=2, gen_on_worker=True)
    with pytest.raises(AssertionError, match="gen_on_worker"):
        QueueService(vocs, sim_specs={}, libE_specs=libE_specs, exit_criteria={"sim_max": 1})


def test_suggest_returns_empty_after_shutdown_seen():
    gen = _make_gen()
    gen.poll_timeout = 0.05
    gen._shutdown_seen = True
    assert gen.suggest(1) == []
    assert gen.suggest(5) == []


def test_suggest_shutdown_sentinel_sets_flag():
    gen = _make_gen()
    gen.poll_timeout = 0.1
    sentinel = QueueGenerator.shutdown_sentinel()
    gen.input_queue.put(sentinel)
    assert gen.suggest(1) == []
    assert gen._shutdown_seen is True
    assert gen.suggest(1) == []


def test_finalize_forwards_results():
    gen = _make_gen()
    gen.finalize([{"x": 1.0, "y": 2.0}])
    assert gen.output_queue.get_nowait() == {"x": 1.0, "y": 2.0}
    gen.finalize(None)
    gen.finalize([])
    assert gen.output_queue.empty()


def test_shutdown_sentinel_identity():
    s1 = QueueGenerator.shutdown_sentinel()
    s2 = QueueGenerator.shutdown_sentinel()
    assert s1 is s2


def test_get_completed_drains():
    vocs = VOCS(variables={"x": [-1, 1]})
    from libensemble.specs import ExitCriteria, LibeSpecs

    service = QueueService(vocs, sim_specs={}, libE_specs=LibeSpecs(nworkers=1), exit_criteria=ExitCriteria(sim_max=1))
    assert service.get_completed() == []
    service.output_queue.put({"x": 1.0})
    service.output_queue.put({"x": 2.0})
    out = service.get_completed()
    assert out == [{"x": 1.0}, {"x": 2.0}]
    assert service.get_completed() == []


def test_collect_results_timeout_empty():
    vocs = VOCS(variables={"x": [-1, 1]})
    from libensemble.specs import ExitCriteria, LibeSpecs

    service = QueueService(vocs, sim_specs={}, libE_specs=LibeSpecs(nworkers=1), exit_criteria=ExitCriteria(sim_max=1))
    start = time.time()
    out = service.collect_results(1, timeout=1.0)
    elapsed = time.time() - start
    assert out == []
    assert 0.9 <= elapsed < 2.0

    service.output_queue.put({"x": 9.0})
    out = service.collect_results(1, timeout=1.0)
    assert out == [{"x": 9.0}]


def test_stream_results_timeout_and_yield():
    vocs = VOCS(variables={"x": [-1, 1]})
    from libensemble.specs import ExitCriteria, LibeSpecs

    service = QueueService(vocs, sim_specs={}, libE_specs=LibeSpecs(nworkers=1), exit_criteria=ExitCriteria(sim_max=1))
    start = time.time()
    out = list(service.stream_results(n=1, timeout=1.0))
    elapsed = time.time() - start
    assert out == []
    assert 0.9 <= elapsed < 2.0

    service.output_queue.put({"x": 1.0})
    service.output_queue.put({"x": 2.0})
    out = list(service.stream_results(n=2, timeout=2.0))
    assert out == [{"x": 1.0}, {"x": 2.0}]


def test_shutdown_wait_joins():
    vocs = VOCS(variables={"x": [-1, 1]})
    from libensemble.specs import ExitCriteria, LibeSpecs

    service = QueueService(vocs, sim_specs={}, libE_specs=LibeSpecs(nworkers=1), exit_criteria=ExitCriteria(sim_max=1))
    mock_thread = MagicMock()
    service._thread = mock_thread
    service.shutdown(wait=True)
    assert service.input_queue.get_nowait() is QueueGenerator.shutdown_sentinel()
    mock_thread.join.assert_called_once()

    mock_thread.reset_mock()
    service.shutdown(wait=False)
    mock_thread.join.assert_not_called()


if __name__ == "__main__":
    test_getstate_strips_queues()
    test_suggest_raises_clear_error_without_queues()
    test_ingest_raises_clear_error_without_queues()
    test_queue_service_rejects_gen_on_worker()
