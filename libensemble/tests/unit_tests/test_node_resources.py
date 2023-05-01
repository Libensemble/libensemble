import os

from libensemble.resources import node_resources
from libensemble.resources.env_resources import EnvResources


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


def test_complete_set():
    assert not node_resources._complete_set([None, None, None])
    assert not node_resources._complete_set([2, None, 5])
    assert not node_resources._complete_set([2, 8, None])
    assert node_resources._complete_set([2, 4, 6])
    assert node_resources._complete_set([2, 0, 5])


def test_cpu_info_complete():
    assert not node_resources._cpu_info_complete([None, None, None])
    assert not node_resources._cpu_info_complete([2, None, 5])
    assert node_resources._cpu_info_complete([2, 8, None])
    assert node_resources._cpu_info_complete([2, 4, 6])


def test_gpu_info_complete():
    assert not node_resources._gpu_info_complete([None, None, None])
    assert node_resources._gpu_info_complete([2, None, 5])
    assert not node_resources._gpu_info_complete([2, 8, None])
    assert node_resources._gpu_info_complete([2, 4, 6])


def test_update_values():
    result = node_resources._update_values([None, 2, 3], [11, 12, 13])
    assert result == [11, 12, 3], f"Unexpected result {result}"
    result = node_resources._update_values([1, 2, None], [11, 12, 13])
    assert result == [1, 2, 13], f"Unexpected result {result}"


def test_update_from_str():
    result = node_resources._update_from_str([None, 2, 3], "11 12 13")
    assert result == [11, 12, 3], f"Unexpected result {result}"
    result = node_resources._update_from_str([1, 2, None], "11 12 13")
    assert result == [1, 2, 13], f"Unexpected result {result}"


if __name__ == "__main__":
    setup_standalone_run()

    test_get_cpu_resources_from_env_empty()
    test_get_cpu_resources_from_env_lsf()
    test_get_cpu_resources_from_env_lsf_shortform()
    test_get_cpu_resources_from_env_unknown_env()
    test_complete_set()
    test_cpu_info_complete()
    test_gpu_info_complete()
    test_update_values()
    test_update_from_str()

    teardown_standalone_run()
