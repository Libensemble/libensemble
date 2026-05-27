"""
Unit tests for Flux Framework integration in libEnsemble.

Tests cover:
- FLUX_MPIRunner command generation
- Flux nodelist parsing (via slurm-style bracket notation)
- Flux MPI variant detection
- FluxAllocation platform configuration
- FluxExecutor (when flux bindings available)
"""

import os
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from libensemble.executors import flux_executor
from libensemble.executors.mpi_runner import FLUX_MPIRunner, MPIRunner
from libensemble.resources.env_resources import EnvResources
from libensemble.resources.platforms import FluxAllocation, Known_platforms
from libensemble.utils import launcher
from libensemble.utils.validators import check_mpi_runner_type

# ========================================================================================
# Tests for FLUX_MPIRunner
# ========================================================================================


def test_flux_runner_factory_registration():
    """Test that flux runner is registered in the factory"""
    runner = MPIRunner.get_runner("flux")
    assert runner is not None
    assert isinstance(runner, FLUX_MPIRunner)


def test_flux_runner_default_command():
    """Test default flux run command"""
    runner = FLUX_MPIRunner()
    assert runner.run_command == "flux"
    assert runner.subgroup_launch is False
    assert runner.mfile_support is False


def test_flux_runner_mpi_command_template():
    """Test the MPI command template for flux"""
    runner = FLUX_MPIRunner()
    expected = ["flux", "run", "-N {num_nodes}", "-n {num_procs}", "{extra_args}"]
    assert runner.mpi_command == expected


def test_flux_runner_forms_valid_runline_without_tasks_per_node():
    """Flux run should avoid mixing per-resource and per-task options"""
    runner = FLUX_MPIRunner()
    specs = runner.get_mpi_specs(
        task=SimpleNamespace(env={}, _add_to_env=lambda *args: None, ngpus_req=0),
        nprocs=4,
        nnodes=2,
        ppn=2,
        ngpus=None,
        machinefile=None,
        hyperthreads=False,
        extra_args=None,
        auto_assign_gpus=False,
        match_procs_to_gpus=False,
        resources=None,
        workerID=1,
    )

    runline = launcher.form_command(runner.mpi_command, specs)
    assert runline == ["flux", "run", "-N", "2", "-n", "4", "-c", "1"]


def test_flux_runner_arg_parsing():
    """Test argument parsing configuration"""
    runner = FLUX_MPIRunner()
    assert "-n" in runner.arg_nprocs
    assert "--ntasks" in runner.arg_nprocs
    assert "-N" in runner.arg_nnodes
    assert "--nodes" in runner.arg_nnodes
    assert "--tasks-per-node" in runner.arg_ppn


def test_flux_runner_gpu_settings():
    """Test GPU argument configuration"""
    runner = FLUX_MPIRunner()
    assert runner.default_gpu_arg_type == "option_gpus_per_task"
    assert runner.default_gpu_args["option_gpus_per_task"] == "-g"
    assert runner.default_gpu_args["option_gpus_per_node"] == "--gpus-per-node"


def test_flux_runner_express_spec():
    """Test that express_spec returns None for both hostlist and machinefile"""
    runner = FLUX_MPIRunner()
    hostlist, machinefile = runner.express_spec(
        task=None,
        nprocs=4,
        nnodes=2,
        ppn=2,
        machinefile=None,
        hyperthreads=False,
        extra_args=None,
        resources=None,
        workerID=1,
    )
    assert hostlist is None
    assert machinefile is None


def test_flux_runner_custom_command():
    """Test custom run command override"""
    runner = FLUX_MPIRunner(run_command="/custom/flux")
    assert runner.run_command == "/custom/flux"
    assert runner.mpi_command[0] == "/custom/flux"


# ========================================================================================
# Tests for Flux nodelist parsing
# ========================================================================================


def test_flux_nodelist_from_string_empty():
    """Test parsing empty nodelist string"""
    result = EnvResources.get_slurm_nodelist_from_string("")
    assert result == []


def test_flux_nodelist_from_string_single():
    """Test parsing single node"""
    result = EnvResources.get_slurm_nodelist_from_string("node001")
    assert result == ["node001"]


def test_flux_nodelist_from_string_range():
    """Test parsing node range (flux uses slurm-style notation)"""
    result = EnvResources.get_slurm_nodelist_from_string("node[001-004]")
    assert result == ["node001", "node002", "node003", "node004"]


def test_flux_nodelist_from_string_mixed():
    """Test parsing mixed single nodes and ranges"""
    result = EnvResources.get_slurm_nodelist_from_string("node[001-002,005],other[010-011]")
    assert "node001" in result
    assert "node002" in result
    assert "node005" in result
    assert "other010" in result
    assert "other011" in result


@mock.patch("subprocess.run")
def test_flux_nodelist_success(mock_run):
    """Test getting nodelist from flux resource list"""
    mock_run.return_value = mock.Mock(returncode=0, stdout="node[001-004]\n", stderr="")

    result = EnvResources.get_flux_nodelist("FLUX_URI")

    mock_run.assert_called_once()
    assert "flux" in mock_run.call_args[0][0]
    assert "resource" in mock_run.call_args[0][0]
    assert result == ["node001", "node002", "node003", "node004"]


@mock.patch("subprocess.run")
def test_flux_nodelist_command_failure(mock_run):
    """Test handling flux command failure"""
    mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error message")

    result = EnvResources.get_flux_nodelist("FLUX_URI")
    assert result == []


@mock.patch("subprocess.run")
def test_flux_nodelist_timeout(mock_run):
    """Test handling flux command timeout"""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="flux", timeout=10)

    result = EnvResources.get_flux_nodelist("FLUX_URI")
    assert result == []


@mock.patch("subprocess.run")
def test_flux_nodelist_not_found(mock_run):
    """Test handling flux command not found"""
    mock_run.side_effect = FileNotFoundError()

    result = EnvResources.get_flux_nodelist("FLUX_URI")
    assert result == []


def test_env_resources_flux_detection():
    """Test that EnvResources detects Flux when FLUX_URI is set"""
    # Save current env
    old_flux_uri = os.environ.get("FLUX_URI")
    old_slurm = os.environ.get("SLURM_NODELIST")

    try:
        # Clear conflicting env vars
        if "SLURM_NODELIST" in os.environ:
            del os.environ["SLURM_NODELIST"]

        # Set FLUX_URI
        os.environ["FLUX_URI"] = "local:///tmp/flux-test"

        env_resources = EnvResources()
        assert env_resources.scheduler == "Flux"
        assert "Flux" in env_resources.nodelists
        assert "Flux" in env_resources.ndlist_funcs

    finally:
        # Restore env
        if old_flux_uri:
            os.environ["FLUX_URI"] = old_flux_uri
        elif "FLUX_URI" in os.environ:
            del os.environ["FLUX_URI"]

        if old_slurm:
            os.environ["SLURM_NODELIST"] = old_slurm


def test_env_resources_flux_env_variable():
    """Test default Flux environment variable"""
    assert EnvResources.default_nodelist_env_flux == "FLUX_URI"


# ========================================================================================
# Tests for Flux MPI variant detection
# ========================================================================================


@mock.patch.dict(os.environ, {"FLUX_URI": "local:///tmp/flux-test"})
@mock.patch("subprocess.check_call")
def test_get_mpi_variant_flux(mock_check_call):
    """Test MPI variant detection returns flux when in Flux instance"""
    from libensemble.resources.mpi_resources import get_MPI_variant

    mock_check_call.return_value = 0  # flux --version succeeds

    result = get_MPI_variant()
    assert result == "flux"

    # Verify flux --version was called
    mock_check_call.assert_called_once()
    assert mock_check_call.call_args[0][0] == ["flux", "--version"]


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("subprocess.check_call")
@mock.patch("subprocess.Popen")
def test_get_mpi_variant_no_flux_uri(mock_popen, mock_check_call):
    """Test MPI variant detection skips flux when FLUX_URI not set"""
    from libensemble.resources.mpi_resources import get_MPI_variant

    # Make all checks fail
    mock_check_call.side_effect = Exception("not found")
    mock_popen.side_effect = FileNotFoundError()

    result = get_MPI_variant()

    # Should not be flux since FLUX_URI not set
    assert result != "flux" or result is None


# ========================================================================================
# Tests for FluxAllocation platform
# ========================================================================================


def test_flux_allocation_platform():
    """Test FluxAllocation platform configuration"""
    platform = FluxAllocation()
    assert platform.mpi_runner == "flux"
    assert platform.runner_name == "flux"
    assert platform.gpu_setting_type == "runner_default"
    assert platform.scheduler_match_slots is False


def test_flux_in_known_platforms():
    """Test that flux is registered in known platforms"""
    platforms = Known_platforms()
    assert hasattr(platforms, "flux")
    assert isinstance(platforms.flux, FluxAllocation)


# ========================================================================================
# Tests for validator
# ========================================================================================


def test_validator_accepts_flux():
    """Test that flux is accepted by the MPI runner validator"""

    class MockCls:
        pass

    result = check_mpi_runner_type(MockCls, "flux")
    assert result == "flux"


def test_validator_accepts_all_runners():
    """Test all valid runner names are accepted"""

    class MockCls:
        pass

    valid_runners = ["mpich", "openmpi", "aprun", "srun", "jsrun", "msmpi", "flux", "custom"]
    for runner in valid_runners:
        result = check_mpi_runner_type(MockCls, runner)
        assert result == runner


def test_validator_rejects_invalid():
    """Test invalid runner names are rejected"""

    class MockCls:
        pass

    with pytest.raises(AssertionError):
        check_mpi_runner_type(MockCls, "invalid_runner")


# ========================================================================================
# Tests for FluxExecutor (conditional on flux availability)
# ========================================================================================


def test_flux_executor_import_without_flux():
    """Test FluxExecutor handles missing flux gracefully"""
    # This test just verifies the module can be imported
    # even when flux is not available
    try:
        from libensemble.executors import flux_executor

        # FLUX_AVAILABLE should be False if flux not installed
        # This is fine - we just want to ensure import doesn't crash
        assert hasattr(flux_executor, "FLUX_AVAILABLE")
    except ImportError:
        pytest.skip("flux_executor module not available")


def test_flux_executor_requires_flux_uri():
    """Test FluxExecutor raises error when FLUX_URI not set"""
    try:
        from libensemble.executors.flux_executor import FLUX_AVAILABLE, FluxExecutor

        if not FLUX_AVAILABLE:
            pytest.skip("Flux Python bindings not available")

        # Save and clear FLUX_URI
        old_uri = os.environ.get("FLUX_URI")
        if "FLUX_URI" in os.environ:
            del os.environ["FLUX_URI"]

        try:
            from libensemble.executors.executor import ExecutorException

            with pytest.raises(ExecutorException, match="FLUX_URI"):
                FluxExecutor()
        finally:
            if old_uri:
                os.environ["FLUX_URI"] = old_uri

    except ImportError:
        pytest.skip("flux_executor module not available")


def test_flux_task_poll_uses_get_job():
    """Test FluxTask polls using Flux's get_job helper"""
    if not flux_executor.FLUX_AVAILABLE:
        pytest.skip("Flux Python bindings not available")

    task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    task.flux_handle = object()
    task.flux_jobid = 123
    task.timer.start()
    task.submit_time = task.timer.tstart

    with mock.patch.object(flux_executor.flux.job, "get_job", return_value={"state": "RUN"}) as mock_get_job:
        task.poll()

    mock_get_job.assert_called_once_with(task.flux_handle, task.flux_jobid)
    assert task.state == "RUNNING"


def test_flux_executor_submit_builds_jobspec_with_environment_and_gpus():
    """Test FluxExecutor submit passes environment and GPU resources via jobspec"""
    if not flux_executor.FLUX_AVAILABLE:
        pytest.skip("Flux Python bindings not available")

    executor = object.__new__(flux_executor.FluxExecutor)
    executor.flux_handle = object()
    executor.resources = None
    executor.platform_info = {}
    executor.workerID = 7
    executor.list_of_tasks = []
    executor.apps = {}
    executor.default_apps = {"sim": None, "gen": None}
    executor.base_dir = os.getcwd()

    app = SimpleNamespace(
        name="sim", full_path="/path/to/sim.x", app_cmd="fluxwrap /path/to/sim.x", precedent="fluxwrap"
    )
    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    old_env = os.environ.get("TEST_FLUX_ENV")
    os.environ["TEST_FLUX_ENV"] = "present"

    jobspec = SimpleNamespace(stdout=None, stderr=None)
    submit_calls = []

    def fake_from_command(command, **kwargs):
        submit_calls.append((command, kwargs))
        jobspec.cwd = kwargs.get("cwd")
        jobspec.environment = kwargs.get("environment")
        jobspec.setattr_shell_option = mock.Mock()
        return jobspec

    try:
        with (
            mock.patch.object(flux_executor.JobspecV1, "from_command", side_effect=fake_from_command),
            mock.patch.object(flux_executor.flux.job, "submit", return_value=42),
        ):
            task = executor.submit(app_name="sim", num_procs=4, num_nodes=2, num_gpus=4, app_args="--flag value")
    finally:
        if old_env is None:
            del os.environ["TEST_FLUX_ENV"]
        else:
            os.environ["TEST_FLUX_ENV"] = old_env

    command, kwargs = submit_calls[0]
    assert command[:2] == ["fluxwrap", "/path/to/sim.x"]
    assert command[-2:] == ["--flag", "value"]
    assert kwargs["num_tasks"] == 4
    assert kwargs["num_nodes"] == 2
    assert kwargs["gpus_per_task"] == 1
    assert kwargs["environment"]["TEST_FLUX_ENV"] == "present"
    assert kwargs["environment"]["LIBENSEMBLE_SIM_DIR"] == "."
    jobspec.setattr_shell_option.assert_called_once_with("gpu-affinity", "per-task")
    assert task.flux_jobid == 42


# ========================================================================================
# Test runner standalone execution
# ========================================================================================


if __name__ == "__main__":
    # FLUX_MPIRunner tests
    test_flux_runner_factory_registration()
    test_flux_runner_default_command()
    test_flux_runner_mpi_command_template()
    test_flux_runner_forms_valid_runline_without_tasks_per_node()
    test_flux_runner_arg_parsing()
    test_flux_runner_gpu_settings()
    test_flux_runner_express_spec()
    test_flux_runner_custom_command()

    # Nodelist parsing tests
    test_flux_nodelist_from_string_empty()
    test_flux_nodelist_from_string_single()
    test_flux_nodelist_from_string_range()
    test_flux_nodelist_from_string_mixed()

    # Validator tests
    test_validator_accepts_flux()
    test_validator_accepts_all_runners()

    # Platform tests
    test_flux_allocation_platform()
    test_flux_in_known_platforms()

    # EnvResources tests
    test_env_resources_flux_env_variable()

    print("All standalone tests passed!")
