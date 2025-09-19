import sys

import numpy as np

import libensemble.tests.unit_tests.setup as setup
from libensemble.utils.misc import specs_dump


def test_ensemble_init():
    """testing init attrs"""
    from libensemble.ensemble import Ensemble

    sys.argv = ["", "--comms", "local", "--nworkers", "4"]

    e = Ensemble(parse_args=True)
    assert hasattr(e.libE_specs, "comms"), "internal parse_args() didn't populate defaults for class's libE_specs"
    assert hasattr(e, "nworkers"), "nworkers should've passed from libE_specs to Ensemble class"
    assert e.is_manager, "parse_args() didn't populate defaults for class's libE_specs"


def test_ensemble_parse_args_false():
    from libensemble.ensemble import Ensemble
    from libensemble.specs import LibeSpecs

    # Ensemble(parse_args=False) by default, so these specs won't be overwritten:
    e = Ensemble(libE_specs={"comms": "local", "nworkers": 4})
    assert hasattr(e, "nworkers"), "nworkers should've passed from libE_specs to Ensemble class"
    assert isinstance(e.libE_specs, LibeSpecs), "libE_specs should've been cast to class"

    # test pass attribute as dict
    e = Ensemble(libE_specs={"comms": "local", "nworkers": 4})
    assert hasattr(e, "nworkers"), "nworkers should've passed from libE_specs to Ensemble class"
    assert isinstance(e.libE_specs, LibeSpecs), "libE_specs should've been cast to class"

    # test that adjusting Ensemble.nworkers also changes libE_specs
    e.nworkers = 8
    assert e.libE_specs.nworkers == 8, "libE_specs nworkers not adjusted"


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
        sim_specs["outputs"] = sim_specs["out"]
        gen_specs["outputs"] = gen_specs["out"]
        sim_specs.pop("in")
        sim_specs.pop("out")
        gen_specs.pop("out")
        assert all([i in specs_dump(e.sim_specs).items() for i in sim_specs.items()])
        assert all([i in specs_dump(e.gen_specs).items() for i in gen_specs.items()])
        assert all([i in specs_dump(e.exit_criteria).items() for i in exit_criteria.items()])


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


def test_full_workflow():
    """Test initializing a workflow via Specs and Ensemble.run()"""
    from libensemble.ensemble import Ensemble
    from libensemble.gen_funcs.sampling import latin_hypercube_sample
    from libensemble.sim_funcs.simple_sim import norm_eval
    from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    LS = LibeSpecs(comms="local", nworkers=4)

    # parameterizes and validates everything!
    ens = Ensemble(
        libE_specs=LS,
        sim_specs=SimSpecs(sim_f=norm_eval),
        gen_specs=GenSpecs(
            gen_f=latin_hypercube_sample,
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
    ens.libE_specs.dry_run = True
    flag = 1
    try:
        ens.run()
    except SystemExit:
        flag = 0
    assert not flag, "Ensemble didn't exit after specifying dry_run"


def test_flakey_workflow():
    """Test initializing a workflow via Specs and Ensemble.run()"""
    from pydantic import ValidationError

    from libensemble.ensemble import Ensemble
    from libensemble.gen_funcs.sampling import latin_hypercube_sample
    from libensemble.sim_funcs.simple_sim import norm_eval
    from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    LS = LibeSpecs(comms="local", nworkers=4)

    flag = 1
    try:
        ens = Ensemble(
            libE_specs=LS,
            sim_specs=SimSpecs(sim_f=norm_eval),
            gen_specs=GenSpecs(
                gen_f=latin_hypercube_sample,
                user={
                    "gen_batch_size": 100,
                    "lb": np.array([-3]),
                    "ub": np.array([3]),
                },
            ),
            exit_criteria=ExitCriteria(gen_max=101),
        )
        ens.sim_specs.inputs = (["x"],)  # note trailing comma
        ens.add_random_streams()
        ens.run()
    except ValidationError:
        flag = 0

    assert not flag, "should've caught input errors"


def test_ensemble_specs_update_libE_specs():
    """Test that libE_specs is updated as expected with .attribute setting"""
    from libensemble.ensemble import Ensemble
    from libensemble.resources.platforms import PerlmutterGPU
    from libensemble.specs import LibeSpecs

    platform_specs = PerlmutterGPU()

    ensemble = Ensemble(
        libE_specs=LibeSpecs(comms="local", nworkers=4),
    )

    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=ensemble.nworkers - 1,
        resource_info={"gpus_on_node": 4},
        use_workflow_dir=True,
        platform_specs=platform_specs,
    )

    assert ensemble.libE_specs.num_resource_sets == ensemble.nworkers - 1
    assert len(str(ensemble.libE_specs.workflow_dir_path)) > 1
    assert ensemble.libE_specs.platform_specs == specs_dump(platform_specs, exclude_none=True)


def test_ensemble_prevent_comms_overwrite():
    """Test that libE_specs is updated as expected with .attribute setting"""
    from libensemble.ensemble import Ensemble
    from libensemble.specs import LibeSpecs

    ensemble = Ensemble(
        libE_specs=LibeSpecs(comms="mpi"),
    )

    flag = 1
    try:
        ensemble.libE_specs = LibeSpecs(comms="local")
    except ValueError:
        flag = 0

    assert not flag, "UserWarning should've been raised upon trying to overwrite comms"

    # test that dot-notation is also disallowed, upon trying .run()
    # TODO: This may not be possible
    flag = 1
    ensemble = Ensemble()
    try:
        ensemble.libE_specs.comms = "mpi"
        ensemble.run()
    except ValueError:
        flag = 0

    assert not flag, "should not be able to overwrite comms with dot-notation"


def test_local_comms_without_nworkers():
    """Test that an empty ensemble can't be created, plus that nworkers must be specified"""
    from libensemble.ensemble import Ensemble
    from libensemble.specs import LibeSpecs

    flag = 1
    try:
        Ensemble(
            libE_specs=LibeSpecs(comms="local"),
        )
    except ValueError:
        flag = 0

    assert not flag, "'local' ensemble without nworkers should not be created"


if __name__ == "__main__":
    test_ensemble_init()
    test_ensemble_parse_args_false()
    test_from_files()
    test_bad_func_loads()
    test_full_workflow()
    test_flakey_workflow()
    test_ensemble_specs_update_libE_specs()
    test_ensemble_prevent_comms_overwrite()
    test_local_comms_without_nworkers()
