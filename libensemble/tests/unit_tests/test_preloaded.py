import numpy as np
import pytest
from gest_api.vocs import VOCS

from libensemble.gen_classes.preloaded import PreloadedSampleGenerator


def test_structured_array_points_are_served_in_requested_chunks():
    points = np.zeros(3, dtype=[("x", float, 2), ("sim_id", int), ("sim_started", bool)])
    points["x"] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    points["sim_id"] = [10, 11, 12]
    generator = PreloadedSampleGenerator(points)

    assert generator.n_remaining == 3
    first_batch = generator.suggest(2)
    np.testing.assert_array_equal([point["x"] for point in first_batch], points["x"][:2])
    assert generator.n_remaining == 1
    final_batch = generator.suggest(2)
    np.testing.assert_array_equal([point["x"] for point in final_batch], points["x"][2:])
    assert generator.suggest(2) == []
    assert generator.n_remaining == 0


def test_list_points_batch_size_and_ingest():
    vocs = VOCS(variables={"x": [0.0, 1.0]})
    points = [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}]
    generator = PreloadedSampleGenerator(points, vocs=vocs, batch_size=2)

    assert generator.suggest(1) == points[:2]
    generator.ingest([{"x": 0.1, "f": 0.01}])
    assert generator.suggest(1) == points[2:]
    assert generator.suggest(1) == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        PreloadedSampleGenerator([], batch_size=batch_size)
