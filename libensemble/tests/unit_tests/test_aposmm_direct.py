"""
Tests for the APOSMMDirect class, which directly implements the APOSMM algorithm
without wrapping the persistent generator function via QCommProcess.
"""

import multiprocessing
import platform

import pytest

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

if platform.system() in ["Linux", "Darwin"]:
    multiprocessing.set_start_method("fork", force=True)

import numpy as np
from gest_api.vocs import VOCS

from libensemble.gen_classes import APOSMMDirect
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima


def _make_vocs_and_mapping():
    """Create standard VOCS and mapping for the six-hump camel problem."""
    variables = {"core": [-3, 3], "edge": [-2, 2], "core_on_cube": [0, 1], "edge_on_cube": [0, 1]}
    objectives = {"energy": "MINIMIZE"}
    variables_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }
    vocs = VOCS(variables=variables, objectives=objectives)
    return vocs, variables_mapping


@pytest.mark.extra
def test_aposmm_direct_suggest_first():
    """Test the suggest-first workflow: suggest initial sample, evaluate, ingest, then optimize."""

    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMMDirect(
        vocs,
        max_active_runs=6,
        initial_sample_size=100,
        variables_mapping=variables_mapping,
        sample_points=np.round(minima, 1),
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
    )

    # Get and evaluate initial sample
    initial_sample = gen.suggest(100)
    assert len(initial_sample) == 100

    total_evals = 0

    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
        total_evals += 1

    gen.ingest(initial_sample)

    # Now run optimization loop
    eval_max = 2000
    potential_minima = []

    while total_evals < eval_max:
        sample = gen.suggest(6)
        detected_minima = gen.suggest_updates()
        if len(detected_minima):
            for m in detected_minima:
                potential_minima.append(m)
        if len(sample) == 0:
            break
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        gen.ingest(sample)

    gen.finalize()
    H, persis_info, exit_code = gen.export()

    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "APOSMMDirect didn't exit correctly"
    assert persis_info.get("run_order"), "APOSMMDirect didn't do any localopt runs"

    assert len(potential_minima) >= 6, f"Found only {len(potential_minima)} minima batches"

    tol = 1e-3
    min_found = 0
    for m in minima:
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 6, f"Found {min_found} minima, expected >= 6"
    print(f"[test_aposmm_direct_suggest_first] Found {min_found} minima")


@pytest.mark.extra
def test_aposmm_direct_ingest_first():
    """Test the ingest-first workflow: provide pre-evaluated sample, then optimize."""

    vocs, variables_mapping = _make_vocs_and_mapping()
    variables = {"core": [-3, 3], "edge": [-2, 2]}

    gen = APOSMMDirect(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
    )

    # Provide pre-evaluated sample
    initial_sample = [
        {
            "core": minima[i][0],
            "edge": minima[i][1],
            "core_on_cube": (minima[i][0] - variables["core"][0]) / (variables["core"][1] - variables["core"][0]),
            "edge_on_cube": (minima[i][1] - variables["edge"][0]) / (variables["edge"][1] - variables["edge"][0]),
            "energy": six_hump_camel_func(np.array([minima[i][0], minima[i][1]])),
        }
        for i in range(6)
    ]
    gen.ingest(initial_sample)

    # Run optimization
    total_evals = 0
    eval_max = 2000
    potential_minima = []

    while total_evals < eval_max:
        sample = gen.suggest(6)
        detected_minima = gen.suggest_updates()
        if len(detected_minima):
            for m in detected_minima:
                potential_minima.append(m)
        if len(sample) == 0:
            break
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        gen.ingest(sample)

    gen.finalize()
    H, persis_info, exit_code = gen.export()

    assert persis_info.get("run_order"), "APOSMMDirect didn't do any localopt runs"
    assert len(potential_minima) >= 4, f"Found only {len(potential_minima)} minima batches"

    tol = 1e-3
    min_found = 0
    for m in minima:
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 4, f"Found {min_found} minima, expected >= 4"
    print(f"[test_aposmm_direct_ingest_first] Found {min_found} minima")


@pytest.mark.extra
def test_aposmm_direct_batched_suggest():
    """Test batched suggest - requesting a subset of available points."""

    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMMDirect(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
    )

    # Suggest in small batches during initial sample
    first = gen.suggest(2)
    assert len(first) == 2

    second = gen.suggest(4)
    assert len(second) == 4

    # Evaluate and ingest all initial sample
    all_points = list(first) + list(second)
    for point in all_points:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(all_points)

    # Run a few optimization iterations
    for _ in range(5):
        sample = gen.suggest(3)
        if len(sample) == 0:
            break
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
        gen.ingest(sample)

    gen.finalize()
    H, persis_info, _ = gen.export()
    assert H is not None
    print(f"[test_aposmm_direct_batched_suggest] Total points in history: {len(H)}")


@pytest.mark.extra
def test_aposmm_direct_stop_after_k_minima():
    """Test early stopping after finding k minima."""

    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMMDirect(
        vocs,
        max_active_runs=6,
        initial_sample_size=100,
        variables_mapping=variables_mapping,
        sample_points=np.round(minima, 1),
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
        stop_after_k_minima=2,
    )

    # Get and evaluate initial sample
    initial_sample = gen.suggest(100)
    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(initial_sample)

    # Run optimization - should stop early
    total_evals = 0
    max_iters = 500

    while total_evals < max_iters:
        sample = gen.suggest(6)
        if len(sample) == 0:
            break
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        gen.ingest(sample)

    gen.finalize()
    H, persis_info, _ = gen.export()
    assert H is not None

    n_min = np.sum(H["local_min"]) if H is not None else 0
    print(f"[test_aposmm_direct_stop_after_k_minima] Found {n_min} minima, stopped after {total_evals} evals")


@pytest.mark.extra
def test_aposmm_direct_validation_errors():
    """Test that validation errors are raised for bad configurations."""

    vocs = VOCS(
        variables={"core": [-3, 3], "edge": [-2, 2], "core_on_cube": [0, 1], "edge_on_cube": [0, 1]},
        objectives={"energy": "MINIMIZE"},
    )

    # Missing x_on_cube mapping
    bad_mapping = {"x": ["core", "edge"], "f": ["energy"]}
    with pytest.raises(ValueError):
        APOSMMDirect(
            vocs,
            max_active_runs=6,
            initial_sample_size=10,
            variables_mapping=bad_mapping,
            localopt_method="scipy_Nelder-Mead",
            opt_return_codes=[0],
        )

    # Mismatched x and x_on_cube sizes
    bad_mapping2 = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube"],
        "f": ["energy"],
    }
    with pytest.raises(ValueError):
        APOSMMDirect(
            vocs,
            max_active_runs=6,
            initial_sample_size=10,
            variables_mapping=bad_mapping2,
            localopt_method="scipy_Nelder-Mead",
            opt_return_codes=[0],
        )


@pytest.mark.extra
def test_aposmm_direct_export():
    """Test export with different options."""

    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMMDirect(
        vocs,
        max_active_runs=6,
        initial_sample_size=10,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
    )

    # Export before any work
    H, _, _ = gen.export()
    assert H is None

    # Suggest, evaluate, ingest
    sample = gen.suggest(10)
    for point in sample:
        point["energy"] = 1.0
    gen.ingest(sample)
    gen.finalize()

    # Export as numpy
    H, persis_info, tag = gen.export()
    assert H is not None
    assert "x" in H.dtype.names
    assert "f" in H.dtype.names

    # Export with vocs field names
    H_unmapped, _, _ = gen.export(vocs_field_names=True)
    if H_unmapped is not None:
        assert "core" in H_unmapped.dtype.names
        assert "edge" in H_unmapped.dtype.names

    # Export as dicts
    H_dicts, _, _ = gen.export(as_dicts=True)
    assert isinstance(H_dicts, list)
    assert isinstance(H_dicts[0], dict)


if __name__ == "__main__":
    test_aposmm_direct_suggest_first()
    test_aposmm_direct_ingest_first()
    test_aposmm_direct_batched_suggest()
    test_aposmm_direct_stop_after_k_minima()
    test_aposmm_direct_validation_errors()
    test_aposmm_direct_export()
