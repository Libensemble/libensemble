import multiprocessing
import platform

import pytest

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

if platform.system() in ["Linux", "Darwin"]:
    multiprocessing.set_start_method("fork", force=True)

import numpy as np

import libensemble.tests.unit_tests.setup as setup
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad

libE_info = {"comm": {}}


@pytest.mark.extra
def test_persis_aposmm_localopt_test():
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    _, _, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(4, dtype=[("f", float), ("sim_id", bool), ("dist_to_unit_bounds", float), ("sim_ended", bool)])
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
    from libensemble.gen_funcs.persistent_aposmm import update_history_optimal

    hist, _, _, _, _ = setup.hist_setup1(n=2)

    H = hist.H

    H["sim_ended"] = True
    H["sim_id"] = range(len(H))
    H["f"][0] = -1e-8
    H["x_on_cube"][-1] = 1e-10

    # Perturb x_opt point to test the case where the reported minimum isn't
    # exactly in H. Also, a point in the neighborhood of x_opt has a better
    # function value.
    opt_ind = update_history_optimal(H["x_on_cube"][-1] + 1e-12, 1, H, np.arange(len(H)))

    assert opt_ind == 9, "Wrong point declared minimum"


def combined_func(x):
    return six_hump_camel_func(x), six_hump_camel_grad(x)


@pytest.mark.extra
def test_standalone_persistent_aposmm():

    import libensemble.gen_funcs
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 4}

    n = 2
    eval_max = 2000

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    gen_specs = {
        "in": ["x", "f", "grad", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            # 'localopt_method': 'LD_MMA', # Needs gradients
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
    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert np.sum(H["sim_ended"]) >= eval_max, "Standalone persistent_aposmm, didn't evaluate enough points"
    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"

    tol = 1e-3
    min_found = 0
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 4, f"Found {min_found} minima"


def _evaluate_aposmm_instance(my_APOSMM, minimum_minima=6):
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    initial_sample = my_APOSMM.suggest(100)

    total_evals = 0
    eval_max = 2000

    for point in initial_sample:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
        total_evals += 1

    my_APOSMM.ingest(initial_sample)

    potential_minima = []

    while total_evals < eval_max:

        sample, detected_minima = my_APOSMM.suggest(6), my_APOSMM.suggest_updates()
        if len(detected_minima):
            for m in detected_minima:
                potential_minima.append(m)
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        my_APOSMM.ingest(sample)
    my_APOSMM.finalize()
    H, persis_info, exit_code = my_APOSMM.export()

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"

    assert len(potential_minima) >= 6, f"Found {len(potential_minima)} minima"

    tol = 1e-3
    min_found = 0
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= minimum_minima, f"Found {min_found} minima"


@pytest.mark.extra
def test_standalone_persistent_aposmm_combined_func():

    import libensemble.gen_funcs
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    from libensemble.gen_funcs.persistent_aposmm import aposmm

    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 4}

    n = 2
    eval_max = 100

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    gen_specs = {
        "in": ["x", "f", "grad", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            # 'localopt_method': 'LD_MMA', # Needs gradients
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

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert np.sum(H["sim_ended"]) >= eval_max, "Standalone persistent_aposmm, didn't evaluate enough points"
    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"


@pytest.mark.extra
def test_asktell_with_persistent_aposmm():

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_APOSMM = APOSMM(
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

    _evaluate_aposmm_instance(my_APOSMM)


@pytest.mark.extra
def test_asktell_errors():

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(
        variables=variables,
        objectives=objectives,
        constraints={"c1": ["LESS_THAN", 0]},
        constants={"alpha": 0.55},
        observables={"o1"},
    )
    with pytest.raises(ValueError):
        APOSMM(
            vocs,
            max_active_runs=6,
            variables_mapping={"x": ["missing"], "f": ["energy"]},
            initial_sample_size=100,
        )

    vocs = VOCS(variables=variables, objectives=objectives)

    my_APOSMM = APOSMM(
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

    my_APOSMM.suggest()
    with pytest.raises(RuntimeError):
        my_APOSMM.suggest()
        pytest.fail("Should've failed on consecutive empty suggests")

    my_APOSMM = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.5,
    )

    with pytest.raises(RuntimeError):
        my_APOSMM.finalize()
        pytest.fail("Should've failed on finalize before start")

    my_APOSMM = APOSMM(
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

    my_APOSMM.suggest()
    with pytest.raises(RuntimeError):
        my_APOSMM.setup()
        pytest.fail("Should've failed on consecutive setup")
    my_APOSMM.finalize()

    from libensemble.utils.runners import Runner

    def gest_style_sim(_):
        return {"energy": 0.0}

    runner = Runner({"sim_f": gest_style_sim})
    with pytest.raises(AttributeError, match="SimSpecs.simulator"):
        runner.run(np.zeros(1), {"persis_info": {}, "libE_info": {}})


@pytest.mark.extra
def test_asktell_ingest_first():
    from math import gamma, pi, sqrt

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

    n = 2

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_APOSMM = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="LN_BOBYQA",
        opt_return_codes=[0],
        rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        xtol_abs=1e-6,
        ftol_abs=1e-6,
        dist_to_bound_multiple=0.01,
    )

    initial_sample = [
        {
            "core": minima[i][0],
            "edge": minima[i][1],
            "energy": six_hump_camel_func(np.array([minima[i][0], minima[i][1]])),
        }
        for i in range(6)
    ]
    my_APOSMM.ingest(initial_sample)

    total_evals = 0
    eval_max = 2000

    potential_minima = []

    while total_evals < eval_max:

        sample, detected_minima = my_APOSMM.suggest(6), my_APOSMM.suggest_updates()
        if len(detected_minima):
            for m in detected_minima:
                potential_minima.append(m)
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        my_APOSMM.ingest(sample)
    my_APOSMM.finalize()
    H, persis_info, exit_code = my_APOSMM.export()

    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"

    assert len(potential_minima) >= 6, f"Found {len(potential_minima)} minima"

    tol = 1e-4
    min_found = 0
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 4, f"Found {min_found} minima"


@pytest.mark.extra
def test_asktell_consecutive_during_sample():
    """Test consecutive suggest and ingest during sample"""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_APOSMM = APOSMM(
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

    # Test consecutive suggest
    first = my_APOSMM.suggest(1)
    first[0]["energy"] = six_hump_camel_func(np.array([first[0]["core"], first[0]["edge"]]))
    my_APOSMM.ingest(first)
    second = my_APOSMM.suggest(1)
    second += my_APOSMM.suggest(4)
    for point in second:
        point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
    # Test consecutive ingest
    my_APOSMM.ingest(second[:3])
    my_APOSMM.ingest(second[3:])

    total_evals = 0
    eval_max = 2000

    potential_minima = []

    while total_evals < eval_max:

        sample, detected_minima = my_APOSMM.suggest(3), my_APOSMM.suggest_updates()
        sample += my_APOSMM.suggest(3)
        if len(detected_minima):
            for m in detected_minima:
                potential_minima.append(m)
        for point in sample:
            point["energy"] = six_hump_camel_func(np.array([point["core"], point["edge"]]))
            total_evals += 1
        my_APOSMM.ingest(sample)

    my_APOSMM.finalize()
    H, persis_info, exit_code = my_APOSMM.export()

    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"

    assert len(potential_minima) >= 6, f"Found {len(potential_minima)} minima"

    tol = 1e-3
    min_found = 0
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 4, f"Found {min_found} minima"


def _run_aposmm_export_test(variables_mapping):
    """Helper function to run APOSMM export tests with given variables_mapping"""
    from gest_api.vocs import VOCS

    from libensemble.gen_classes import APOSMM
    from libensemble.specs import GenSpecs

    variables = {
        "core": [-3, 3],
        "edge": [-2, 2],
    }
    objectives = {"energy": "MINIMIZE"}

    vocs = VOCS(variables=variables, objectives=objectives)
    aposmm = APOSMM(
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
    specs = GenSpecs(generator=aposmm, vocs=vocs)
    assert "x_on_cube" not in [field[0] for field in specs.outputs]
    assert "x_on_cube" not in specs.persis_in
    H, _, _ = aposmm.export()
    assert H is None  # Should be None before finalize
    # Test export after suggest/ingest cycle
    sample = aposmm.suggest(5)
    for point in sample:
        point["energy"] = 1.0  # Mock evaluation
    aposmm.ingest(sample)
    aposmm.finalize()

    # Test export with unmapped fields
    H, _, _ = aposmm.export()
    if H is not None:
        assert "x" in H.dtype.names and H["x"].ndim == 2
        assert "x_on_cube" in H.dtype.names and H["x_on_cube"].ndim == 2
        assert "f" in H.dtype.names and H["f"].ndim == 1

    # Test export with vocs_field_names
    H_unmapped, _, _ = aposmm.export(vocs_field_names=True)
    print(f"H_unmapped: {H_unmapped}")  # Debug
    if H_unmapped is not None:
        assert "core" in H_unmapped.dtype.names
        assert "edge" in H_unmapped.dtype.names
        assert "energy" in H_unmapped.dtype.names
    # Test export with as_dicts
    H_dicts, _, _ = aposmm.export(as_dicts=True)
    assert isinstance(H_dicts, list)
    assert isinstance(H_dicts[0], dict)
    assert "x" in H_dicts[0]  # x remains as array
    assert "f" in H_dicts[0]
    # Test export with both options
    H_both, _, _ = aposmm.export(vocs_field_names=True, as_dicts=True)
    assert isinstance(H_both, list)
    assert "core" in H_both[0]
    assert "edge" in H_both[0]
    assert "energy" in H_both[0]


@pytest.mark.extra
def test_aposmm_export():
    """Test APOSMM export function with different options"""

    mapping = {"x": ["core", "edge"], "f": ["energy"]}
    _run_aposmm_export_test(mapping)


@pytest.mark.extra
def test_aposmm_no_x_mapping():
    """Test APOSMM raises ValueError when no variables mapped to 'x'."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    vocs = VOCS(variables=variables, objectives=objectives)

    with pytest.raises(ValueError, match="requires at least one variable mapped to 'x'"):
        APOSMM(
            vocs,
            max_active_runs=6,
            initial_sample_size=6,
            variables_mapping={"x": [], "f": ["energy"]},
        )


@pytest.mark.extra
def test_aposmm_components_in_kwargs():
    """Test that 'components' in kwargs adds 'fvec' to persis_in."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        components=2,
    )
    assert "fvec" in my_aposmm.gen_specs["persis_in"]


@pytest.mark.extra
def test_aposmm_remove_internal_coordinates_none():
    """Test _remove_internal_coordinates handles None input."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
    )
    assert my_aposmm._remove_internal_coordinates(None) is None


@pytest.mark.extra
def test_aposmm_remove_internal_coordinates_no_fields():
    """Test _remove_internal_coordinates returns same array when no internal fields to remove."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
    )
    dtype = [("x", float, 2), ("f", float)]
    results = np.zeros(3, dtype=dtype)
    output = my_aposmm._remove_internal_coordinates(results)
    assert output is results


@pytest.mark.extra
def test_aposmm_add_internal_coordinates_none_or_empty():
    """Test _add_internal_coordinates handles None or empty input."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
    )
    assert my_aposmm._add_internal_coordinates(None) is None
    assert my_aposmm._add_internal_coordinates(np.array([])) is not None


@pytest.mark.extra
def test_aposmm_add_internal_coordinates_missing_x():
    """Test _add_internal_coordinates raises ValueError when 'x' missing after mapping."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping={"f": ["energy"]},
    )
    dtype = [("energy", float)]
    results = np.zeros(3, dtype=dtype)
    with pytest.raises(ValueError, match="must include the 'x' variable"):
        my_aposmm._add_internal_coordinates(results)


@pytest.mark.extra
def test_aposmm_add_internal_coordinates_with_internal_vocs_fields():
    """Test _add_internal_coordinates removes internal vocs fields during ingest."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
        "x_on_cube": ["extra"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
    )
    # Results with already-mapped 'x' and 'f', plus an internal vocs field 'extra'
    dtype = [("x", float, 2), ("f", float), ("extra", float, 2)]
    results = np.zeros(3, dtype=dtype)
    results["x"] = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    results["f"] = [1.0, 2.0, 3.0]
    results["extra"] = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

    output = my_aposmm._add_internal_coordinates(results)
    # The 'extra' field should be stripped, 'x_on_cube' should be added
    assert "extra" not in output.dtype.names
    assert "x_on_cube" in output.dtype.names


@pytest.mark.extra
def test_aposmm_internal_slot_in_data_with_id():
    """Test _slot_in_data with _id field handling."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=10,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
    )
    dtype = [("_id", int), ("core", float), ("edge", float), ("energy", float)]
    results = np.zeros(3, dtype=dtype)
    results["_id"] = [10, 11, 12]
    results["core"] = [-1.0, 0.0, 1.0]
    results["edge"] = [-1.0, 0.0, 1.0]
    results["energy"] = [1.0, 2.0, 3.0]

    # Set up the ingest buffer
    my_aposmm._ingest_buf = np.zeros(10, dtype=[("sim_id", int), ("core", float), ("edge", float), ("energy", float)])
    my_aposmm._slot_in_data(results)
    assert list(my_aposmm._ingest_buf["sim_id"][:3]) == [10, 11, 12]


@pytest.mark.extra
def test_aposmm_periodic_vocs():
    """Test APOSMM with periodic variables."""

    from gest_api.vocs import VOCS

    import libensemble.gen_funcs
    from libensemble.gen_classes import APOSMM

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    variables = {"core": [-3, 3], "edge": [-2, 2]}
    objectives = {"energy": "MINIMIZE"}

    variables_mapping = {
        "x": ["core", "edge"],
        "f": ["energy"],
    }

    vocs = VOCS(variables=variables, objectives=objectives)

    my_aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=6,
        variables_mapping=variables_mapping,
        localopt_method="scipy_Nelder-Mead",
        periodic=True,
    )
    dtype = [("core", float), ("edge", float), ("energy", float)]
    results = np.zeros(3, dtype=dtype)
    results["core"] = [-3.0, 0.0, 3.0]
    results["edge"] = [-2.0, 0.0, 2.0]
    results["energy"] = [1.0, 2.0, 3.0]

    output = my_aposmm._add_internal_coordinates(results)
    # x_on_cube should be wrapped due to periodic
    assert "x_on_cube" in output.dtype.names
    # core=3.0 maps to x_on_cube=1.0, then wrapped to 0.0 with periodic
    assert output["x_on_cube"][2][0] == 0.0


if __name__ == "__main__":
    test_persis_aposmm_localopt_test()
    test_update_history_optimal()
    test_standalone_persistent_aposmm()
    test_standalone_persistent_aposmm_combined_func()
    test_asktell_with_persistent_aposmm()
    test_asktell_ingest_first()
    test_asktell_consecutive_during_sample()
    test_asktell_errors()
    test_aposmm_export()
    test_aposmm_no_x_mapping()
    test_aposmm_components_in_kwargs()
    test_aposmm_remove_internal_coordinates_none()
    test_aposmm_remove_internal_coordinates_no_fields()
    test_aposmm_add_internal_coordinates_none_or_empty()
    test_aposmm_add_internal_coordinates_missing_x()
    test_aposmm_add_internal_coordinates_with_internal_vocs_fields()
    test_aposmm_internal_slot_in_data_with_id()
    test_aposmm_periodic_vocs()
