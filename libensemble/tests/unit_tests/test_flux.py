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
from libensemble.executors.executor import TimeoutExpired
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


def test_flux_executor_connects_lazily_with_default_or_explicit_uri():
    """FluxExecutor should defer Flux connection and support explicit URIs."""
    try:
        from libensemble.executors.flux_executor import FLUX_AVAILABLE, FluxExecutor

        if not FLUX_AVAILABLE:
            pytest.skip("Flux Python bindings not available")

        with mock.patch.object(flux_executor.flux, "Flux", return_value="default-handle") as mock_flux:
            executor = FluxExecutor()
            assert executor.flux_handle is None
            assert executor._get_flux_handle() == "default-handle"
            mock_flux.assert_called_once_with()

        with mock.patch.object(flux_executor.flux, "Flux", return_value="uri-handle") as mock_flux:
            executor = FluxExecutor(uri="local:///tmp/flux-uri")
            assert executor._get_flux_handle() == "uri-handle"
            mock_flux.assert_called_once_with("local:///tmp/flux-uri")

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


def _make_uninitialized_flux_executor(*, worker_id: int = 7):
    executor = object.__new__(flux_executor.FluxExecutor)
    executor.flux_handle = object()
    executor.platform_info = {}
    executor.uri = None
    executor.workerID = worker_id
    executor.list_of_tasks = []
    executor.apps = {}
    executor.default_apps = {"sim": None, "gen": None}
    executor.base_dir = os.getcwd()
    return executor


def test_flux_executor_submit_builds_jobspec_with_environment_and_gpus():
    """Test FluxExecutor submit passes environment and GPU resources via jobspec"""

    executor = _make_uninitialized_flux_executor(worker_id=7)

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
    submit_future = SimpleNamespace(get_id=mock.Mock(return_value=42))

    def fake_from_command(command, **kwargs):
        submit_calls.append((command, kwargs))
        jobspec.cwd = kwargs.get("cwd")
        jobspec.environment = kwargs.get("environment")
        jobspec.setattr_shell_option = mock.Mock()
        return jobspec

    try:
        with (
            mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
            mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
            mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
            mock.patch.object(flux_executor.flux.job, "submit_async", return_value=submit_future) as mock_submit_async,
        ):
            _patched_jobspecV1.from_command.side_effect = fake_from_command
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
    assert kwargs["output"] == os.path.join(task.workdir, task.stdout)
    assert kwargs["error"] == os.path.join(task.workdir, task.stderr)
    mock_submit_async.assert_called_once_with(executor.flux_handle, jobspec)
    submit_future.get_id.assert_called_once_with()
    jobspec.setattr_shell_option.assert_called_once_with("gpu-affinity", "per-task")
    assert task.flux_jobid == 42
    assert task.flux_future is submit_future


def test_flux_executor_submit_dry_run_marks_task_complete_and_does_not_submit_job():
    """Dry-run should not call JobspecV1.from_command or flux.job.submit_async."""
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = mock.Mock()

    mock_from_command = mock.Mock()

    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(flux_executor.flux.job, "submit_async") as mock_submit,
    ):
        _patched_jobspecV1.from_command = mock_from_command
        task = executor.submit(
            app_name="sim",
            num_procs=2,
            num_nodes=1,
            num_gpus=0,
            app_args="--flag",
            stdout="out.txt",
            stderr="err.txt",
            dry_run=True,
        )

    mock_from_command.assert_not_called()
    mock_submit.assert_not_called()
    executor._check_app_exists.assert_not_called()
    assert task.finished is True
    assert task.state == "FINISHED"
    assert task.success is True
    assert task.runline is not None
    assert len(executor.list_of_tasks) == 1


def test_flux_executor_submit_wait_on_start_invokes_waiter():
    """wait_on_start should call FluxExecutor._wait_on_start when not in dry_run."""
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    jobspec = SimpleNamespace(stdout=None, stderr=None)
    submit_future = SimpleNamespace(get_id=mock.Mock(return_value=123))

    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(flux_executor.flux.job, "submit_async", return_value=submit_future),
        mock.patch.object(executor, "_wait_on_start") as mock_wait_on_start,
    ):
        _patched_jobspecV1.from_command.return_value = jobspec
        task = executor.submit(
            app_name="sim",
            num_procs=2,
            num_nodes=1,
            num_gpus=None,
            wait_on_start=7,
        )

    mock_wait_on_start.assert_called_once_with(task, 7)
    assert task.flux_jobid == 123


def test_flux_executor_submit_validation_errors():
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    # Missing app_name + calc_type
    with pytest.raises(Exception):
        executor.submit()

    # extra_args unsupported
    with pytest.raises(Exception, match="extra_args"):
        executor.submit(app_name="sim", num_procs=1, num_nodes=1, extra_args="--x")

    # num_gpus negative
    with pytest.raises(Exception, match="num_gpus must be non-negative"):
        executor.submit(app_name="sim", num_procs=2, num_nodes=1, num_gpus=-1)

    # num_gpus not divisible by num_procs
    with pytest.raises(Exception, match="num_gpus must be divisible by num_procs"):
        executor.submit(app_name="sim", num_procs=3, num_nodes=1, num_gpus=2)

    # procs_per_node divides num_procs -> num_nodes inferred
    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(
            flux_executor.flux.job, "submit_async", return_value=SimpleNamespace(get_id=mock.Mock(return_value=1))
        ),
    ):
        _patched_jobspecV1.from_command.return_value = SimpleNamespace(stdout=None, stderr=None)
        executor.submit(app_name="sim", num_procs=4, procs_per_node=2)

    # num_procs must be divisible by procs_per_node
    with pytest.raises(Exception, match="divisible by procs_per_node"):
        executor.submit(app_name="sim", num_procs=3, procs_per_node=2)

    # num_procs must equal num_nodes * procs_per_node
    with pytest.raises(Exception, match=r"num_procs must equal num_nodes \* procs_per_node"):
        executor.submit(app_name="sim", num_procs=5, num_nodes=2, procs_per_node=3)


def test_flux_executor_submit_error_from_flux_job_submit_sets_failed_to_start():
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    jobspec = SimpleNamespace(stdout=None, stderr=None)

    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(flux_executor.flux.job, "submit_async", side_effect=RuntimeError("boom")),
    ):
        _patched_jobspecV1.from_command.return_value = jobspec
        with pytest.raises(Exception, match="Failed to submit Flux job"):
            executor.submit(app_name="sim", num_procs=2, num_nodes=1)

    # Submit failed; the task is created, but it may or may not be appended
    # depending on where the exception is raised. Ensure the exception type is correct.
    assert executor.list_of_tasks == []


def test_flux_executor_submit_stdout_stderr_are_placed_under_workdir():
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    submit_calls = []

    def fake_from_command(command, **kwargs):
        submit_calls.append(kwargs)
        return SimpleNamespace(stdout=None, stderr=None)

    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(
            flux_executor.flux.job, "submit_async", return_value=SimpleNamespace(get_id=mock.Mock(return_value=1))
        ),
    ):
        _patched_jobspecV1.from_command.side_effect = fake_from_command
        task = executor.submit(
            app_name="sim",
            num_procs=2,
            num_nodes=1,
            stdout="my_stdout.txt",
            stderr="my_stderr.txt",
        )

    assert task.flux_jobid == 1
    assert submit_calls[0]["output"].endswith(os.path.join(task.workdir, "my_stdout.txt"))
    assert submit_calls[0]["error"].endswith(os.path.join(task.workdir, "my_stderr.txt"))


def test_flux_executor_submit_sets_gpu_affinity_only_when_num_gpus_nonzero():
    executor = _make_uninitialized_flux_executor(worker_id=7)

    app = SimpleNamespace(
        name="sim",
        full_path="/path/to/sim.x",
        app_cmd="fluxwrap /path/to/sim.x",
        precedent="fluxwrap",
    )

    executor.get_app = lambda app_name: app
    executor.default_app = lambda calc_type: app
    executor._check_app_exists = lambda app_obj: None

    jobspec = SimpleNamespace(stdout=None, stderr=None)
    jobspec.setattr_shell_option = mock.Mock()

    with (
        mock.patch.object(flux_executor, "JobspecV1", create=True, autospec=False) as _patched_jobspecV1,
        mock.patch.object(flux_executor, "flux", create=True) as _patched_flux,  # noqa: F841
        mock.patch.object(flux_executor.flux, "job", create=True) as _patched_job,  # noqa: F841
        mock.patch.object(
            flux_executor.flux.job, "submit_async", return_value=SimpleNamespace(get_id=mock.Mock(return_value=1))
        ),
    ):
        _patched_jobspecV1.from_command.return_value = jobspec
        executor.submit(app_name="sim", num_procs=4, num_nodes=1, num_gpus=0)

    jobspec.setattr_shell_option.assert_not_called()


def test_flux_executor_getstate_drops_flux_handle():
    """FluxExecutor should not serialize an open Flux handle into worker processes."""
    with mock.patch.object(flux_executor, "FLUX_AVAILABLE", True):
        executor = flux_executor.FluxExecutor(uri="local:///tmp/flux-test")

    executor.flux_handle = "flux-handle"
    state = executor.__getstate__()
    assert state["flux_handle"] is None
    assert executor.flux_handle == "flux-handle"


def test_flux_executor_wait_on_start_polls_until_running():
    """Test FluxExecutor waits for a FluxTask to leave the startup states."""
    executor = object.__new__(flux_executor.FluxExecutor)
    task = SimpleNamespace(
        name="flux-task",
        state="CREATED",
        finished=False,
        timer=SimpleNamespace(tstart=None, start=mock.Mock(side_effect=lambda: setattr(task.timer, "tstart", 1.23))),
        submit_time=None,
    )

    def poll_side_effect():
        task.state = "RUNNING"

    task.poll = mock.Mock(side_effect=poll_side_effect)

    with mock.patch.object(flux_executor.time, "sleep"):
        executor._wait_on_start(task, timeout=0.5)

    assert task.poll.call_count == 1
    assert task.state == "RUNNING"
    assert task.timer.start.call_count == 2
    assert task.submit_time == 1.23


def test_flux_task_poll_maps_completion_waiting_and_unknown_states():
    """Test FluxTask poll maps Flux job states to libEnsemble states."""
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
    fake_flux = SimpleNamespace(job=SimpleNamespace(get_job=mock.Mock()))

    with mock.patch.object(flux_executor, "flux", fake_flux):
        fake_flux.job.get_job.return_value = {"state": "SCHED"}
        task.poll()
        assert task.state == "WAITING"
        assert not task.finished

        with mock.patch.object(task, "_handle_completion") as mock_handle_completion:
            fake_flux.job.get_job.return_value = {"state": "INACTIVE"}
            task.poll()
        mock_handle_completion.assert_called_once_with({"state": "INACTIVE"})

        fake_flux.job.get_job.return_value = {"state": "MYSTERY"}
        task.finished = False
        task.poll()

    assert task.state == "UNKNOWN"


def test_flux_task_handle_completion_success_and_failure():
    """Test FluxTask completion handling sets success, state, and errcode."""
    success_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    success_task.timer.start()
    success_task.submit_time = success_task.timer.tstart
    success_task._handle_completion({"state": "INACTIVE", "result": "COMPLETED", "returncode": 0})
    assert success_task.finished is True
    assert success_task.success is True
    assert success_task.state == "FINISHED"
    assert success_task.errcode == 0

    failed_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    failed_task.timer.start()
    failed_task.submit_time = failed_task.timer.tstart
    failed_task._handle_completion({"state": "INACTIVE", "result": "FAILED", "returncode": 7})
    assert failed_task.finished is True
    assert failed_task.success is False
    assert failed_task.state == "FAILED"
    assert failed_task.errcode == 7


def test_flux_task_set_complete_handles_dry_run_and_return_codes():
    """Test FluxTask _set_complete for dry-run and non-dry-run tasks."""
    dry_run_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=True,
    )
    dry_run_task._set_complete()
    assert dry_run_task.finished is True
    assert dry_run_task.success is True
    assert dry_run_task.state == "FINISHED"

    finished_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    finished_task.errcode = 3
    finished_task.timer.start()
    finished_task.submit_time = finished_task.timer.tstart
    finished_task._set_complete()
    assert finished_task.finished is True
    assert finished_task.success is False
    assert finished_task.state == "FAILED"

    # cover waiting on a task that completes before timeout
    task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    task._set_complete()
    task.flux_jobid = 123
    task.wait(timeout=10)
    task.kill()


def test_flux_task_dry_run_exception_and_kill():
    """Test FluxTask dry run exception attributes."""
    task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=True,
    )
    task.wait()
    assert task.finished is True
    assert task.success is True
    assert task.state == "FINISHED"
    task.kill()
    task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=True,
    )
    task.poll()
    assert task.finished is True
    assert task.success is True
    assert task.state == "FINISHED"


def test_flux_task_wait_completes_and_times_out():
    """Test FluxTask wait completes after polling and raises on timeout."""
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

    def complete_on_second_poll():
        complete_on_second_poll.calls += 1
        if complete_on_second_poll.calls == 1:
            task.state = "RUNNING"
        else:
            task.finished = True
            task.state = "FINISHED"

    complete_on_second_poll.calls = 0
    task.poll = mock.Mock(side_effect=complete_on_second_poll)

    with mock.patch.object(flux_executor.time, "sleep"):
        task.wait(timeout=1.0)

    assert task.finished is True
    assert task.state == "FINISHED"
    assert task.poll.call_count == 2

    result_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    result_task.flux_handle = object()
    result_task.flux_jobid = 321
    result_task.timer.start()
    result_task.submit_time = result_task.timer.tstart
    result_info = SimpleNamespace(result="COMPLETED", returncode=0)
    fake_flux = SimpleNamespace(job=SimpleNamespace(result=mock.Mock(return_value=result_info)))

    with mock.patch.object(flux_executor, "flux", fake_flux):
        result_task.wait()

    fake_flux.job.result.assert_called_once_with(result_task.flux_handle, result_task.flux_jobid)
    assert result_task.finished is True
    assert result_task.state == "FINISHED"

    timeout_task = flux_executor.FluxTask(
        app=SimpleNamespace(name="app"),
        app_args=None,
        workdir=os.getcwd(),
        stdout="out.txt",
        stderr="err.txt",
        workerid=1,
        dry_run=False,
    )
    timeout_task.flux_handle = object()
    timeout_task.flux_jobid = 456
    timeout_task.poll = mock.Mock(side_effect=lambda: setattr(timeout_task, "state", "RUNNING"))

    with (
        mock.patch.object(flux_executor.time, "sleep"),
        mock.patch.object(flux_executor.time, "time", side_effect=[0.0, 0.2]),
    ):
        with pytest.raises(TimeoutExpired):
            timeout_task.wait(timeout=0.1)


def test_flux_task_kill_cancels_and_marks_user_killed():
    """Test FluxTask kill cancels the job and marks the task as user-killed."""
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
    task.flux_jobid = 789
    task.timer.start()
    task.submit_time = task.timer.tstart

    def poll_side_effect():
        if poll_side_effect.calls == 0:
            task.state = "RUNNING"
        else:
            task.finished = True
            task.state = "FAILED"
        poll_side_effect.calls += 1

    poll_side_effect.calls = 0
    task.poll = mock.Mock(side_effect=poll_side_effect)
    fake_flux = SimpleNamespace(job=SimpleNamespace(cancel=mock.Mock()))

    with (
        mock.patch.object(flux_executor, "flux", fake_flux),
        mock.patch.object(flux_executor.time, "sleep"),
        mock.patch.object(flux_executor.time, "time", side_effect=[0.0, 0.0, 0.2]),
    ):
        task.kill(wait_time=1)

    fake_flux.job.cancel.assert_called_once_with(task.flux_handle, task.flux_jobid)
    assert task.state == "USER_KILLED"
    assert task.finished is True


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
    test_validator_rejects_invalid()

    # Platform tests
    test_flux_allocation_platform()
    test_flux_in_known_platforms()

    # EnvResources tests
    test_env_resources_flux_env_variable()

    # Flux Executor tests
    test_flux_executor_getstate_drops_flux_handle()
    test_flux_executor_wait_on_start_polls_until_running()
    test_flux_task_poll_maps_completion_waiting_and_unknown_states()
    test_flux_task_handle_completion_success_and_failure()
    test_flux_task_set_complete_handles_dry_run_and_return_codes()
    test_flux_task_dry_run_exception_and_kill()
    test_flux_task_wait_completes_and_times_out()
    test_flux_task_kill_cancels_and_marks_user_killed()

    print("All standalone tests passed!")
