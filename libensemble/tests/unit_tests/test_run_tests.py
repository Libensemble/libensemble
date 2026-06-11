from argparse import Namespace

import pytest

from libensemble.tests import run_tests


def make_args(**overrides):
    values = {
        "a": None,
        "coverage": True,
        "e": False,
        "exclude_match": None,
        "feature": None,
        "match": None,
        "tier": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parse_test_directives(tmp_path):
    script = tmp_path / "test_example.py"
    script.write_text(
        "# TESTSUITE_COMMS: mpi local\n"
        "# TESTSUITE_NPROCS: 2 5\n"
        "# TESTSUITE_EXTRA: true\n"
        "# TESTSUITE_OS_SKIP: OSX\n"
        "# TESTSUITE_TIER: slow\n"
        "# TESTSUITE_FEATURES: external redis\n"
    )

    directives = run_tests.parse_test_directives(script, largest_nprocs_only=True)

    assert directives == {
        "comms": ["mpi", "local"],
        "nprocs": [5],
        "extra": True,
        "exclude": False,
        "os_skip": ["OSX"],
        "ompi_skip": False,
        "tier": "slow",
        "features": ["external", "redis"],
    }


@pytest.mark.parametrize(
    "directive",
    [
        "# TESTSUITE_COMMS: invalid",
        "# TESTSUITE_NPROCS: 0",
        "# TESTSUITE_EXTRA: yes",
        "# TESTSUITE_OS_SKIP: PLAN9",
        "# TESTSUITE_TIER: overnight",
    ],
)
def test_parse_test_directives_rejects_invalid_values(tmp_path, directive):
    script = tmp_path / "test_invalid.py"
    script.write_text(f"{directive}\n")

    with pytest.raises(ValueError):
        run_tests.parse_test_directives(script)


def test_make_run_line_without_coverage():
    args = make_args(coverage=False)

    cmd = run_tests.make_run_line(["python"], "test_example.py", "local", 4, args)

    assert cmd == ["python", "-W", "ignore::DeprecationWarning", "test_example.py", "--comms", "local", "--nworkers", "3"]


def test_make_run_line_with_mpi_coverage():
    args = make_args(a="--oversubscribe")

    cmd = run_tests.make_run_line(["python"], "test_example.py", "mpi", 4, args)

    assert cmd[:4] == ["mpiexec", "-np", "4", "--oversubscribe"]
    assert cmd[4:] == ["python", "-W", "ignore::DeprecationWarning", *run_tests.cov_opts, "test_example.py"]


def test_skip_test_applies_tier_feature_and_name_filters():
    directives = {
        "extra": False,
        "exclude": False,
        "os_skip": [],
        "tier": "external",
        "features": ["redis", "transport"],
    }
    args = make_args(tier=["external"], feature=["redis"], match=["proxy"])

    assert not run_tests.skip_test(directives, args, "LIN", "/tests/test_proxystore.py")
    assert run_tests.skip_test(directives, args, "LIN", "/tests/test_other.py")

    args.exclude_match = ["proxystore"]
    assert run_tests.skip_test(directives, args, "LIN", "/tests/test_proxystore.py")
