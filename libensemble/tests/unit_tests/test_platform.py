import pytest

from libensemble.resources.platforms import PlatformException, get_platform, known_system_detect

my_spec = {
    "mpi_runner": "srun",
    "gpus_per_node": 4,
    "cores_per_node": 32,
}

summit_spec = {
    "mpi_runner": "jsrun",
    "runner_name": None,
    "cores_per_node": 42,
    "logical_cores_per_node": 168,
    "gpus_per_node": 6,
    "gpu_setting_type": "option_gpus_per_task",
    "gpu_setting_name": "-g",
    "scheduler_match_slots": False,
}


def test_platform_empty():
    """Test no platform options supplied"""
    exp = {}
    libE_specs = {}
    platform_info = get_platform(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"

    libE_specs = {"platform_specs": {}}
    platform_info = get_platform(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


def test_unknown_platform():
    """Test unknown platform supplied"""
    libE_specs = {"platform": "dontexist"}

    with pytest.raises(PlatformException):
        platform_info = get_platform(libE_specs)  # noqa
        pytest.fail("Expected PlatformException")


def test_platform_known():
    """Test known platform supplied"""
    exp = {
        "mpi_runner": "mpich",
        "gpu_setting_type": "env",
        "gpu_setting_name": "ROCR_VISIBLE_DEVICES",
        "scheduler_match_slots": True,
    }

    libE_specs = {"platform": "generic_rocm"}
    platform_info = get_platform(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


def test_platform_specs():
    """Test known platform and platform_specs supplied"""
    exp = my_spec
    libE_specs = {"platform_specs": my_spec}
    platform_info = get_platform(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"

    exp = {
        "mpi_runner": "srun",
        "gpu_setting_type": "env",
        "gpu_setting_name": "ROCR_VISIBLE_DEVICES",
        "scheduler_match_slots": True,
        "gpus_per_node": 4,
        "cores_per_node": 32,
    }
    libE_specs = {"platform": "generic_rocm", "platform_specs": my_spec}
    platform_info = get_platform(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


def test_known_sys_detect():
    get_sys_cmd = "echo summit.olcf.ornl.gov"  # Overrides default "hostname -d"
    platform_info = known_system_detect(cmd=get_sys_cmd)
    assert platform_info == summit_spec, f"Summit spec does not match expected ({platform_info})"

    # Try unknown system
    get_sys_cmd = "echo madeup.system"  # Overrides default "hostname -d"
    platform_info = known_system_detect(cmd=get_sys_cmd)
    assert (
        platform_info == {}
    ), f"Expected known_system_detect to return empty dict for unknown system ({platform_info})"


if __name__ == "__main__":
    test_platform_empty()
    test_unknown_platform()
    test_platform_known()
    test_platform_specs()
    test_known_sys_detect()
