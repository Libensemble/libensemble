from queue import Queue

import pytest
from gest_api.vocs import VOCS

from libensemble.gen_classes.queue_gen import QueueGenerator, QueueService
from libensemble.specs import LibeSpecs


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
    vocs = VOCS(variables={"x": [-1, 1]})
    libE_specs = LibeSpecs(nworkers=2, gen_on_worker=True)
    with pytest.raises(AssertionError, match="gen_on_worker"):
        QueueService(vocs, sim_specs={}, libE_specs=libE_specs, exit_criteria={"sim_max": 1})


if __name__ == "__main__":
    test_getstate_strips_queues()
    test_suggest_raises_clear_error_without_queues()
    test_ingest_raises_clear_error_without_queues()
    test_queue_service_rejects_gen_on_worker()
