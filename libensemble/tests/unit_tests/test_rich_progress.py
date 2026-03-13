"""Unit tests for RichProgress live data class."""

import pytest


class MockHist:
    """Minimal mock for libensemble History used in tests."""

    def __init__(self, sim_ended_count=0, gen_informed_count=0):
        self.sim_ended_count = sim_ended_count
        self.gen_informed_count = gen_informed_count


def test_rich_progress_sim_max():
    """RichProgress tracks sim_ended_count when sim_max is set."""
    pytest.importorskip("rich")
    from libensemble.tools.live_data.rich_progress import RichProgress

    exit_criteria = {"sim_max": 10}
    rp = RichProgress(exit_criteria)
    assert rp._sim_max == 10
    assert rp._total == 10
    assert rp._description == "Sims completed"

    hist = MockHist(sim_ended_count=5)
    rp.live_update(hist)
    task = rp._progress.tasks[rp._task_id]
    assert task.completed == 5

    rp.finalize(hist)


def test_rich_progress_gen_max():
    """RichProgress tracks gen_informed_count when gen_max is set."""
    pytest.importorskip("rich")
    from libensemble.tools.live_data.rich_progress import RichProgress

    exit_criteria = {"gen_max": 20}
    rp = RichProgress(exit_criteria)
    assert rp._gen_max == 20
    assert rp._total == 20
    assert rp._description == "Gen points informed"

    hist = MockHist(gen_informed_count=7)
    rp.live_update(hist)
    task = rp._progress.tasks[rp._task_id]
    assert task.completed == 7

    rp.finalize(hist)


def test_rich_progress_no_criteria():
    """RichProgress works without exit_criteria (unbounded spinner)."""
    pytest.importorskip("rich")
    from libensemble.tools.live_data.rich_progress import RichProgress

    rp = RichProgress()
    assert rp._total is None
    assert rp._description == "Running"

    hist = MockHist(sim_ended_count=3)
    rp.live_update(hist)
    task = rp._progress.tasks[rp._task_id]
    assert task.completed == 3

    rp.finalize(hist)


def test_rich_progress_sim_max_priority():
    """sim_max takes priority over gen_max when both are set."""
    pytest.importorskip("rich")
    from libensemble.tools.live_data.rich_progress import RichProgress

    exit_criteria = {"sim_max": 10, "gen_max": 20}
    rp = RichProgress(exit_criteria)
    assert rp._sim_max == 10
    assert rp._total == 10
    assert rp._description == "Sims completed"

    rp.finalize(MockHist())


def test_rich_progress_pydantic_exit_criteria():
    """RichProgress works with pydantic ExitCriteria object."""
    pytest.importorskip("rich")
    from libensemble.specs import ExitCriteria
    from libensemble.tools.live_data.rich_progress import RichProgress

    exit_criteria = ExitCriteria(sim_max=50)
    rp = RichProgress(exit_criteria)
    assert rp._sim_max == 50
    assert rp._total == 50

    rp.finalize(MockHist())
