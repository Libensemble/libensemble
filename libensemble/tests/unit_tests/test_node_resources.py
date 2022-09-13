import os
from libensemble.resources.env_resources import EnvResources
from libensemble.resources import node_resources


def setup_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def teardown_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def setup_function(function):
    print(f"setup_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def teardown_function(function):
    print(f"teardown_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


# Tests ========================================================================================


def test_get_cpu_resources_from_env_empty():
    # Test empty call
    cores_info = node_resources._get_cpu_resources_from_env()
    assert cores_info is None, "cores_info should be None"


def test_get_cpu_resources_from_env_lsf():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5" + " g06n02" * 42
    exp_out = (42, 42)

    env_resources1 = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    cores_info = node_resources._get_cpu_resources_from_env(env_resources=env_resources1)
    assert cores_info == exp_out, "cores_info returned does not match expected"

    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5" + " g06n02" * 42 + " h21n18" * 42
    env_resources2 = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    cores_info = node_resources._get_cpu_resources_from_env(env_resources=env_resources2)
    assert cores_info == exp_out, "cores_info returned does not match expected"


def test_get_cpu_resources_from_env_lsf_shortform():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5 1 g06n02 42"
    exp_out = (42, 42)

    env_resources1 = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="LIBE_RESOURCES_TEST_NODE_LIST",
    )
    cores_info = node_resources._get_cpu_resources_from_env(env_resources=env_resources1)
    assert cores_info == exp_out, "cores_info returned does not match expected"

    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5 1 g06n02 42 h21n18 42"
    env_resources2 = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="LIBE_RESOURCES_TEST_NODE_LIST",
    )
    cores_info = node_resources._get_cpu_resources_from_env(env_resources=env_resources2)
    assert cores_info == exp_out, "cores_info returned does not match expected"


def test_get_cpu_resources_from_env_unknown_env():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0009-0012]"
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )

    cores_info = node_resources._get_cpu_resources_from_env(env_resources=env_resources)
    assert cores_info is None, "cores_info should be None"


if __name__ == "__main__":
    setup_standalone_run()

    test_get_cpu_resources_from_env_empty()
    test_get_cpu_resources_from_env_lsf()
    test_get_cpu_resources_from_env_lsf_shortform()
    test_get_cpu_resources_from_env_unknown_env()

    teardown_standalone_run()
