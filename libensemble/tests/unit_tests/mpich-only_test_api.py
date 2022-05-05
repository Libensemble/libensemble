import pytest
import numpy as np
import pprint
from libensemble.version import __version__
import libensemble.tests.unit_tests.setup as setup


@pytest.mark.extra
def test_ensemble_init():
    """Only testing attrs most likely to encounter errors"""
    from libensemble.api import Ensemble

    e = Ensemble()
    assert "comms" in e.libE_specs, "parse_args() didn't populate default value for class instance's libE_specs"

    assert e.logger.get_level() == 20, "Default log level should be 20."

    assert e._filename == __file__, "Class instance's detected calling script not correctly set."


@pytest.mark.extra
def test_from_yaml():
    """Test that Ensemble() specs dicts resemble setup dicts"""
    from libensemble.api import Ensemble

    e = Ensemble()
    e.from_yaml("./simdir/test_example.yaml")

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    # unit_test sample specs dont (but perhaps should) contain 'user' field
    sim_specs.update({"user": {}})

    assert all([e.alloc_specs[i] is not None for i in ["user"]]), "class instance's alloc_specs wasn't populated."

    assert e.exit_criteria == exit_criteria, "exit_criteria wasn't correctly loaded or separated from libE_specs."

    assert e.sim_specs == sim_specs, "instance's sim_specs isn't equivalent to sample sim_specs"

    # can't specify np arrays in yaml - have to manually update.
    e.gen_specs["user"]["ub"] = np.ones(1)
    e.gen_specs["user"]["lb"] = np.zeros(1)

    assert e.gen_specs == gen_specs, "instance's gen_specs isn't equivalent to sample gen_specs"


@pytest.mark.extra
def test_str_rep():
    """Test that Ensemble() string rep resembles setup dicts string reps"""
    from libensemble.api import Ensemble

    e = Ensemble()
    e.from_yaml("./simdir/test_example.yaml")

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    e.gen_specs["user"]["ub"] = np.ones(1)
    e.gen_specs["user"]["lb"] = np.zeros(1)

    repd = str(e).split()

    unmatched_strings = []

    for spec in [sim_specs, gen_specs, exit_criteria]:
        for i in pprint.pformat(spec).split():
            if i in repd:
                continue
            else:
                unmatched_strings.append(i)

    # Only possible unmatches should be addresses?
    assert all(["0x" in item for item in unmatched_strings]), "String representation components didn't match expected."

    assert __version__ in repd, "libEnsemble version not detected in string representation."


@pytest.mark.extra
def test_bad_func_loads():
    """Test that Ensemble() raises expected errors (with warnings) on incorrect imports"""
    from libensemble.api import Ensemble

    yaml_errors = {
        "./simdir/test_example_badfuncs_attribute.yaml": AttributeError,
        "./simdir/test_example_badfuncs_notfound.yaml": ModuleNotFoundError,
    }

    for f in yaml_errors:
        e = Ensemble()
        flag = 1
        try:
            e.from_yaml(f)
        except yaml_errors[f]:
            flag = 0
        assert flag == 0


if __name__ == "__main__":
    test_ensemble_init()
    test_from_yaml()
    test_str_rep()
    test_bad_func_loads()
