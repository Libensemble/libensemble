import pytest
from libensemble.resources.platforms import GPU_SET_DEF
from libensemble.resources.platforms import get_platform_from_specs, PlatformException

my_spec = {"mpi_runner": "srun",
          "gpus_per_node": 4,
          "cores_per_node": 32,
          }

def test_platform_empy():
    """Test no platform options supplied"""
    exp = {}
    libE_specs = {}
    platform_info = get_platform_from_specs(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"

    libE_specs = {"platform_specs": {}}
    platform_info = get_platform_from_specs(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


def test_unknown_platform():
    """Test unknown platform supplied"""
    libE_specs = {"platform": 'dontexist'}

    with pytest.raises(PlatformException):
        platform_info = get_platform_from_specs(libE_specs)
        pytest.fail("Expected PlatformException")


def test_platform_known():
    """Test known platform supplied"""
    exp = {'mpi_runner': 'mpich',
        'gpu_setting_type': 2,
        'gpu_setting_name': 'ROCR_VISIBLE_DEVICES',
        'scheduler_match_slots': True}

    libE_specs = {"platform": 'generic_rocm'}
    platform_info = get_platform_from_specs(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


def test_platform_specs():
    """Test known platform and platform_specs supplied"""
    exp = my_spec
    libE_specs = {"platform_specs": my_spec}
    platform_info = get_platform_from_specs(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"

    exp = {"mpi_runner": "srun",
        'gpu_setting_type': 2,
        'gpu_setting_name': 'ROCR_VISIBLE_DEVICES',
        'scheduler_match_slots': True,
        "gpus_per_node": 4,
        "cores_per_node": 32,
        }
    libE_specs = {"platform": 'generic_rocm', "platform_specs": my_spec}
    platform_info = get_platform_from_specs(libE_specs)
    assert platform_info == exp, f"platform_info does not match expected: {platform_info}"


if __name__ == "__main__":
    test_platform_empy()
    test_unknown_platform()
    test_platform_known()
    test_platform_specs()
