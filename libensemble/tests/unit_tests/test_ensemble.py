import numpy as np
import pytest

import libensemble.tests.unit_tests.setup as setup


@pytest.mark.extra
def test_ensemble_init():
    """testing init attrs"""
    from libensemble.ensemble import Ensemble

    e = Ensemble(
        libE_specs={"comms": "local", "nworkers": 4}
    )  # without specifying, class assumes MPI since pytest runs without --comms local
    assert "comms" in e.libE_specs, "internal parse_args() didn't populate defaults for class's libE_specs"
    assert e.is_manager, "parse_args() didn't populate defaults for class's libE_specs"

    assert e.logger.get_level() == 20, "Default log level should be 20."


@pytest.mark.extra
def test_from_files():
    """Test that Ensemble() specs dicts resemble setup dicts"""
    from libensemble.ensemble import Ensemble

    for ft in ["yaml", "json", "toml"]:
        e = Ensemble(libE_specs={"comms": "local", "nworkers": 4})
        file_path = f"./simdir/test_example.{ft}"
        if ft == "yaml":
            e.from_yaml(file_path)
        elif ft == "json":
            e.from_json(file_path)
        else:
            e.from_toml(file_path)

        sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

        e.gen_specs.user["ub"] = np.ones(1)
        e.gen_specs.user["lb"] = np.zeros(1)

        sim_specs["inputs"] = sim_specs["in"]
        sim_specs.pop("in")
        assert all([i in e.sim_specs.__dict__.items() for i in sim_specs.items()])
        assert all([i in e.gen_specs.__dict__.items() for i in gen_specs.items()])
        assert all([i in e.exit_criteria.__dict__.items() for i in exit_criteria.items()])


@pytest.mark.extra
def test_bad_func_loads():
    """Test that Ensemble() raises expected errors (with warnings) on incorrect imports"""
    from libensemble.ensemble import Ensemble

    yaml_errors = {
        "./simdir/test_example_badfuncs_attribute.yaml": AttributeError,
        "./simdir/test_example_badfuncs_notfound.yaml": ModuleNotFoundError,
    }

    for f in yaml_errors:
        e = Ensemble(libE_specs={"comms": "local", "nworkers": 4})
        flag = 1
        try:
            e.from_yaml(f)
        except yaml_errors[f]:
            flag = 0
        assert flag == 0


@pytest.mark.extra
def test_full_workflow():
    """Test initializing a workflow via Specs and Ensemble.run()"""
    from libensemble.ensemble import Ensemble
    from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    LS = LibeSpecs(comms="local", nworkers=4)

    # parameterizes and validates everything!
    ens = Ensemble(
        libE_specs=LS,
        sim_specs=SimSpecs(inputs=["x"], out=[("f", float)]),
        gen_specs=GenSpecs(
            out=[("x", float, (1,))],
            user={
                "gen_batch_size": 100,
                "lb": np.array([-3]),
                "ub": np.array([3]),
            },
        ),
        exit_criteria=ExitCriteria(gen_max=101),
    )
    ens.add_random_streams()
    ens.run()
    if ens.is_manager:
        assert len(ens.H) >= 101

    # test a dry run
    LS = LibeSpecs(comms="local", nworkers=4, dry_run=True)
    ens.libE_specs = LS
    flag = 1
    try:
        ens.run()
    except SystemExit:
        flag = 0
    assert not flag, "Ensemble didn't exit after specifying dry_run"


if __name__ == "__main__":
    test_ensemble_init()
    test_from_files()
    test_bad_func_loads()
    test_full_workflow()
