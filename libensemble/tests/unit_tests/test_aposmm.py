"""
Unit tests for APOSMM generators.

Covers:
- Legacy standalone mode (aposmm gen_func called directly)
- New ask-tell interface via APOSMM (direct implementation)
- APOSMMLegacy (QCommProcess wrapper) compatibility
- Export, validation, and edge-case tests
"""

import multiprocessing
import platform

import numpy as np
import pytest
from gest_api.vocs import VOCS

import libensemble.gen_funcs
from libensemble.sim_funcs.six_hump_camel import (
    six_hump_camel_func,
    six_hump_camel_grad,
)
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

if platform.system() in ["Linux", "Darwin"]:
    multiprocessing.set_start_method("fork", force=True)


libE_info = {"comm": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocs_and_mapping():
    """Standard VOCS and mapping for the six-hump camel problem."""
    variables = {
        "core": [-3, 3],
        "edge": [-2, 2],
        "core_on_cube": [0, 1],
        "edge_on_cube": [0, 1],
    }
    objectives = {"energy": "MINIMIZE"}
    variables_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }
    vocs = VOCS(variables=variables, objectives=objectives)
    return vocs, variables_mapping


def combined_func(x):
    return six_hump_camel_func(x), six_hump_camel_grad(x)


def _run_aposmm_loop(gen, eval_max=2000, batch=6):
    """Run the standard suggest/evaluate/ingest loop on an initialized APOSMM gen.
    Returns (H, persis_info, exit_code, potential_minima).
    """
    total_evals = 0
    potential_minima = []

    while total_evals < eval_max:
        sample = gen.suggest(batch)
        detected_minima = gen.suggest_updates()
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
    return H, persis_info, exit_code, potential_minima


def _assert_minima_found(H, min_count=6, tol=1e-3):
    found = 0
    for m in minima:
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            found += 1
    assert found >= min_count, f"Found {found} minima, expected >= {min_count}"
    return found


# ---------------------------------------------------------------------------
# Legacy gen_func tests (standalone mode -- aposmm called directly)
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_persis_aposmm_localopt_test():
    """Unknown local optimizer method raises NotImplementedError."""
    import libensemble.tests.unit_tests.setup as setup
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    _, _, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(
        4,
        dtype=[
            ("f", float),
            ("sim_id", bool),
            ("dist_to_unit_bounds", float),
            ("sim_ended", bool),
        ],
    )
    H["sim_ended"] = True
    H["sim_id"] = range(len(H))
    gen_specs_0["user"]["localopt_method"] = "BADNAME"
    gen_specs_0["user"]["ub"] = np.ones(2)
    gen_specs_0["user"]["lb"] = np.zeros(2)

    try:
        aposmm(H, {}, gen_specs_0, libE_info)
    except NotImplementedError:
        assert 1, "Failed because method is unknown."
    else:
        assert 0


@pytest.mark.extra
def test_update_history_optimal():
    """update_history_optimal handles x_opt not exactly in H."""
    import libensemble.tests.unit_tests.setup as setup
    from libensemble.gen_funcs.persistent_aposmm import update_history_optimal

    hist, _, _, _, _ = setup.hist_setup1(n=2)
    H = hist.H
    H["sim_ended"] = True
    H["sim_id"] = range(len(H))
    H["f"][0] = -1e-8
    H["x_on_cube"][-1] = 1e-10

    opt_ind = update_history_optimal(H["x_on_cube"][-1] + 1e-12, 1, H, np.arange(len(H)))
    assert opt_ind == 9, "Wrong point declared minimum"


@pytest.mark.extra
def test_standalone_persistent_aposmm():
    """Standalone mode: aposmm gen_func evaluates six-hump camel and finds all 6 minima."""
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 4}
    n = 2
    eval_max = 2000

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "in": [
            "x",
            "f",
            "grad",
            "local_pt",
            "sim_id",
            "sim_ended",
            "x_on_cube",
            "local_min",
        ],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "sample_points": np.round(minima, 1),
            "localopt_method": "scipy_Nelder-Mead",
            "standalone": {
                "eval_max": eval_max,
                "obj_func": six_hump_camel_func,
                "grad_func": six_hump_camel_grad,
            },
            "opt_return_codes": [0],
            "nu": 1e-8,
            "mu": 1e-8,
            "dist_to_bound_multiple": 0.01,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }
    H = []
    H, persis_info, exit_code = aposmm(H, persis_info, gen_specs, libE_info)
    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
    assert np.sum(H["sim_ended"]) >= eval_max
    assert persis_info.get("run_order")
    _assert_minima_found(H)


@pytest.mark.extra
def test_standalone_persistent_aposmm_combined_func():
    """Standalone mode with combined obj_and_grad_func."""
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    n = 2
    eval_max = 100

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "in": [
            "x",
            "f",
            "grad",
            "local_pt",
            "sim_id",
            "sim_ended",
            "x_on_cube",
            "local_min",
        ],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "sample_points": np.round(minima, 1),
            "localopt_method": "scipy_Nelder-Mead",
            "standalone": {"eval_max": eval_max, "obj_and_grad_func": combined_func},
            "opt_return_codes": [0],
            "nu": 1e-8,
            "mu": 1e-8,
            "dist_to_bound_multiple": 0.01,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    H = []
    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 3}
    H, persis_info, exit_code = aposmm(H, persis_info, gen_specs, libE_info)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
    assert np.sum(H["sim_ended"]) >= eval_max
    assert persis_info.get("run_order")


# ---------------------------------------------------------------------------
# APOSMM (direct implementation) -- suggest-first workflow
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_suggest_first():
    """Suggest-first workflow: suggest initial sample, evaluate, ingest, then optimize."""
    from libensemble.gen_classes import APOSMM
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMM(
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

    initial_sample = gen.suggest(100)
    assert len(initial_sample) == 100
    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(initial_sample)

    H, persis_info, exit_code, potential_minima = _run_aposmm_loop(gen)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
    assert persis_info.get("run_order")
    assert len(potential_minima) >= 6
    _assert_minima_found(H)


# ---------------------------------------------------------------------------
# APOSMM (direct implementation) -- ingest-first workflow
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_ingest_first():
    """Ingest-first workflow with NLopt LN_BOBYQA."""
    from math import gamma, pi, sqrt

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

    n = 2
    variables = {
        "core": [-3, 3],
        "edge": [-2, 2],
        "core_on_cube": [0, 1],
        "edge_on_cube": [0, 1],
    }
    objectives = {"energy": "MINIMIZE"}
    variables_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }
    vocs = VOCS(variables=variables, objectives=objectives)

    gen = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="LN_BOBYQA",
        rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        xtol_abs=1e-6,
        ftol_abs=1e-6,
        dist_to_bound_multiple=0.01,
    )

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

    H, persis_info, exit_code, potential_minima = _run_aposmm_loop(gen)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
    assert persis_info.get("run_order")
    assert len(potential_minima) >= 6
    tol = 1e-4
    found = 0
    for m in minima:
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            found += 1
    assert found >= 4, f"Found {found} minima"


# ---------------------------------------------------------------------------
# APOSMM (direct implementation) -- batched suggest
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_batched_suggest():
    """Batched suggest during initial sample phase."""
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMM(
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

    first = gen.suggest(2)
    assert len(first) == 2

    second = gen.suggest(4)
    assert len(second) == 4

    all_points = list(first) + list(second)
    for point in all_points:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(all_points)

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


# ---------------------------------------------------------------------------
# APOSMM (direct implementation) -- stop_after_k_minima
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_stop_after_k_minima():
    """Early stopping after finding k minima."""
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMM(
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

    initial_sample = gen.suggest(100)
    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(initial_sample)

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


# ---------------------------------------------------------------------------
# APOSMM (direct implementation) -- consecutive suggest/ingest during sample
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_consecutive_during_sample():
    """Consecutive suggest and ingest calls during the initial sample phase."""
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMM(
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

    # Consecutive suggest
    first = gen.suggest(1)
    first[0]["energy"] = six_hump_camel_func(np.array([first[0]["core"], first[0]["edge"]]))
    gen.ingest(first)
    second = gen.suggest(1)
    second += gen.suggest(4)
    for point in second:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    # Consecutive ingest
    gen.ingest(second[:3])
    gen.ingest(second[3:])

    total_evals = 0
    eval_max = 2000
    potential_minima = []

    while total_evals < eval_max:
        sample = gen.suggest(3)
        sample += gen.suggest(3)
        detected_minima = gen.suggest_updates()
        for m in detected_minima:
            potential_minima.append(m)
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        gen.ingest(sample)

    gen.finalize()
    H, persis_info, _ = gen.export()

    assert persis_info.get("run_order")
    assert len(potential_minima) >= 6

    found = 0
    tol = 1e-3
    for m in minima:
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            found += 1
    assert found >= 4, f"Found {found} minima"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_validation_errors():
    """Validation errors for bad variable mappings."""
    from libensemble.gen_classes import APOSMM

    vocs = VOCS(
        variables={
            "core": [-3, 3],
            "edge": [-2, 2],
            "core_on_cube": [0, 1],
            "edge_on_cube": [0, 1],
        },
        objectives={"energy": "MINIMIZE"},
        constraints={"c1": ["LESS_THAN", 0]},
        constants={"alpha": 0.55},
    )

    # Missing x_on_cube
    with pytest.raises(ValueError):
        APOSMM(
            vocs,
            max_active_runs=6,
            initial_sample_size=10,
            variables_mapping={"x": ["core", "edge"], "f": ["energy"]},
            localopt_method="scipy_Nelder-Mead",
        )

    # Mismatched x and x_on_cube sizes
    with pytest.raises(ValueError):
        APOSMM(
            vocs,
            max_active_runs=6,
            initial_sample_size=10,
            variables_mapping={
                "x": ["core", "edge"],
                "x_on_cube": ["core_on_cube"],
                "f": ["energy"],
            },
            localopt_method="scipy_Nelder-Mead",
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _run_aposmm_export_test(variables_mapping):
    """Helper: run APOSMM export tests with the given variables_mapping."""
    from libensemble.gen_classes import APOSMM

    variables = {
        "core": [-3, 3],
        "edge": [-2, 2],
        "core_on_cube": [0, 1],
        "edge_on_cube": [0, 1],
    }
    objectives = {"energy": "MINIMIZE"}
    vocs = VOCS(variables=variables, objectives=objectives)

    gen = APOSMM(
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

    # Export before any work should return (None, None, None)
    H, _, _ = gen.export()
    assert H is None

    sample = gen.suggest(5)
    for point in sample:
        point["energy"] = 1.0
    gen.ingest(sample)
    gen.finalize()

    # Export as numpy
    H, _, _ = gen.export()
    if H is not None:
        assert "x" in H.dtype.names and H["x"].ndim == 2
        assert "f" in H.dtype.names and H["f"].ndim == 1

    # Export with vocs_field_names
    H_unmapped, _, _ = gen.export(vocs_field_names=True)
    if H_unmapped is not None:
        assert "core" in H_unmapped.dtype.names
        assert "edge" in H_unmapped.dtype.names
        assert "energy" in H_unmapped.dtype.names

    # Export as dicts
    H_dicts, _, _ = gen.export(as_dicts=True)
    assert isinstance(H_dicts, list)
    assert isinstance(H_dicts[0], dict)
    assert "x" in H_dicts[0]
    assert "f" in H_dicts[0]

    # Export with both options
    H_both, _, _ = gen.export(vocs_field_names=True, as_dicts=True)
    assert isinstance(H_both, list)
    assert "core" in H_both[0]
    assert "edge" in H_both[0]
    assert "energy" in H_both[0]


@pytest.mark.extra
def test_aposmm_export():
    """APOSMM export with different option combinations."""
    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    full_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }
    _run_aposmm_export_test(full_mapping)

    minimal_mapping = {"x_on_cube": ["core_on_cube", "edge_on_cube"]}
    _run_aposmm_export_test(minimal_mapping)


# ---------------------------------------------------------------------------
# APOSMMLegacy (QCommProcess wrapper) compatibility
# ---------------------------------------------------------------------------


@pytest.mark.extra
def test_aposmm_legacy_suggest_first():
    """APOSMMLegacy still works as the QCommProcess-based wrapper."""
    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMMLegacy
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    vocs, variables_mapping = _make_vocs_and_mapping()

    gen = APOSMMLegacy(
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

    initial_sample = gen.suggest(100)
    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    gen.ingest(initial_sample)

    H, persis_info, exit_code, potential_minima = _run_aposmm_loop(gen)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
    assert persis_info.get("run_order")
    assert len(potential_minima) >= 6
    _assert_minima_found(H)


@pytest.mark.extra
def test_aposmm_legacy_errors():
    """APOSMMLegacy raises on bad mappings, consecutive empty suggests, and bad setup order."""
    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMMLegacy

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {
        "core": [-3, 3],
        "edge": [-2, 2],
        "core_on_cube": [0, 1],
        "edge_on_cube": [0, 1],
    }
    objectives = {"energy": "MINIMIZE"}
    vocs = VOCS(
        variables=variables,
        objectives=objectives,
        constraints={"c1": ["LESS_THAN", 0]},
        constants={"alpha": 0.55},
    )
    variables_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }

    with pytest.raises(ValueError):
        APOSMMLegacy(
            vocs,
            max_active_runs=6,
            variables_mapping={"x": ["core", "edge"], "f": ["energy"]},
            initial_sample_size=100,
            localopt_method="scipy_Nelder-Mead",
        )

    vocs_clean = VOCS(variables=variables, objectives=objectives)
    gen = APOSMMLegacy(
        vocs_clean,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
    )
    gen.suggest()
    with pytest.raises(RuntimeError):
        gen.suggest()

    gen2 = APOSMMLegacy(
        vocs_clean,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
    )
    with pytest.raises(RuntimeError):
        gen2.finalize()

    gen3 = APOSMMLegacy(
        vocs_clean,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
    )
    gen3.suggest()
    with pytest.raises(RuntimeError):
        gen3.setup()
    gen3.finalize()

    from libensemble.utils.runners import Runner

    def gest_style_sim(_):
        return {"energy": 0.0}

    runner = Runner({"sim_f": gest_style_sim})
    with pytest.raises(AttributeError, match="SimSpecs.simulator"):
        runner.run(np.zeros(1), {"persis_info": {}, "libE_info": {}})


if __name__ == "__main__":
    test_persis_aposmm_localopt_test()
    test_update_history_optimal()
    test_standalone_persistent_aposmm()
    test_standalone_persistent_aposmm_combined_func()
    test_aposmm_suggest_first()
    test_aposmm_ingest_first()
    test_aposmm_batched_suggest()
    test_aposmm_stop_after_k_minima()
    test_aposmm_consecutive_during_sample()
    test_aposmm_validation_errors()
    test_aposmm_export()
    test_aposmm_legacy_suggest_first()
    test_aposmm_legacy_errors()
