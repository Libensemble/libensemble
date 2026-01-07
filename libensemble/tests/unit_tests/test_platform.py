import pytest

from libensemble.resources.platforms import (
    Known_platforms,
    PlatformException,
    get_platform,
    known_envs,
    known_system_detect,
)
from libensemble.utils.misc import specs_dump

my_spec = {
    "mpi_runner": "srun",
    "gpus_per_node": 4,
    "cores_per_node": 32,
}

frontier_spec = {
    "mpi_runner": "srun",
    "cores_per_node": 64,
    "logical_cores_per_node": 128,
    "gpus_per_node": 8,
    "gpu_setting_type": "runner_default",
    "gpu_env_fallback": "ROCR_VISIBLE_DEVICES",
    "scheduler_match_slots": False,
}


def test_platform_empty(monkeypatch):
    """Test no platform options supplied"""

    # Ensure NERSC_HOST not set
    monkeypatch.delenv("NERSC_HOST", raising=False)

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


def test_platform_specs(monkeypatch):
    """Test known platform and platform_specs supplied"""
    from libensemble.specs import LibeSpecs

    # Ensure NERSC_HOST not set
    monkeypatch.delenv("NERSC_HOST", raising=False)

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

    LS = LibeSpecs(platform_specs=exp)
    assert not isinstance(LS.platform_specs, dict), "Internal platform_specs not cast to class"
    assert specs_dump(LS.platform_specs, exclude_none=True) == exp, "Conversion isn't as expected"


def test_known_sys_detect(monkeypatch):
    """Test detection of known system"""

    # Ensure NERSC_HOST not set
    monkeypatch.delenv("NERSC_HOST", raising=False)

    known_platforms = specs_dump(Known_platforms(), exclude_none=True)
    get_sys_cmd = "echo frontier.olcf.ornl.gov"  # Overrides default "hostname -d"
    name = known_system_detect(cmd=get_sys_cmd)
    platform_info = known_platforms[name]
    assert platform_info == frontier_spec, f"Frontier spec does not match expected ({platform_info})"

    # Try unknown system
    get_sys_cmd = "echo madeup.system"  # Overrides default "hostname -d"
    name = known_system_detect(cmd=get_sys_cmd)
    assert name is None, f"Expected known_system_detect to return None ({name})"


def test_env_sys_detect(monkeypatch):
    """Test detection of system partitions"""
    monkeypatch.setenv("NERSC_HOST", "other_host")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu_test_partition")
    name = known_envs()
    assert name is None
    monkeypatch.setenv("NERSC_HOST", "perlmutter")

    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu_test_partition")
    name = known_envs()
    assert name == "perlmutter_g"

    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu_test_partition")
    name = known_envs()
    assert name == "perlmutter_c"

    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)
    name = known_envs()
    assert name == "perlmutter"


if __name__ == "__main__":
    pytest.main([__file__])
