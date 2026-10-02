# Executors and simulation files

Choose the executor based on the application, independently of the generator.

- `Executor`: serial/threaded local subprocess; no MPI launcher.
- `MPIExecutor`: an MPI or resource-aware application launched within assigned resources.

## Serial external application fragment

Combine this fragment with the complete calling-script pattern in
`canonical-patterns.md`; `vocs` and `gen_specs` come from that pattern.

```python
from pathlib import Path
import numpy as np

from libensemble import Ensemble
from libensemble.executors import Executor
from libensemble.executors.executor import TimeoutExpired


def run_external(inputs: dict, libE_info: dict, **kwargs) -> dict:
    executor = libE_info["executor"]
    task = executor.submit(
        app_name="my_app",
        app_args=f"--x {inputs['x']}",
        stdout="app.out",
        stderr="app.err",
    )
    try:
        task.wait(timeout=600)
    except TimeoutExpired:
        task.kill()
        raise
    if not task.success:
        detail = task.read_stderr() if task.stderr_exists() else ""
        raise RuntimeError(
            f"Application failed: state={task.state}, rc={task.errcode}: {detail}"
        )
    value = float(task.read_stdout().strip())
    if not np.isfinite(value):
        raise ValueError(f"Application returned non-finite objective: {value}")
    return {"f": value}


app_path = Path("/REPLACE/WITH/APP").resolve()
if not app_path.is_file():
    raise FileNotFoundError(app_path)
executor = Executor()
executor.register_app(full_path=str(app_path), app_name="my_app")

ensemble = Ensemble(
    executor=executor,
    sim_specs=SimSpecs(simulator=run_external, vocs=vocs),
    gen_specs=gen_specs,
    libE_specs=LibeSpecs(
        comms="local",
        nworkers=4,
        safe_mode=True,
        sim_dirs_make=True,
        ensemble_dir_path="ensemble",
    ),
)
```

`app_args` is split into tokens as an argument string, not interpreted as a shell script.
Do not rely on pipes, redirects, shell expansion, or shell quoting to preserve embedded
whitespace. Put complex or whitespace-bearing values in an input file when possible.
Prefer explicit stdout/stderr filenames and application-specific parsing.
A `wait()` timeout does not kill the process automatically, so kill it before re-raising.
For manager cancellation responsiveness, use `executor.polling_loop(...,
poll_manager=True)` and require its result to represent successful completion.

A standardized dict simulator cannot directly return a legacy calculation-status value.
Choose and document one failure policy:

- **Fail fast:** raise as above. With default exception handling, one bad evaluation may
  terminate the ensemble.
- **Fault tolerant:** after terminating the task, return NaN objective(s) plus a declared
  user status output. Use this only when the generator tolerates non-finite observations,
  and exclude failed status/NaN rows during analysis.

Do not turn a failed process into a plausible objective. For cancellation-responsive
execution, `executor.polling_loop(task, timeout=..., poll_manager=True)` returns a status;
compare it with `WORKER_DONE` from `libensemble.message_numbers` and apply the chosen policy
for every other status.

## MPI application

Setup differs only in executor construction and optional submit resources:

```python
from libensemble.executors import MPIExecutor

executor = MPIExecutor()
executor.register_app(full_path=str(app_path), app_name="my_mpi_app")

task = libE_info["executor"].submit(
    app_name="my_mpi_app",
    app_args="--input input.dat",
    stdout="app.out",
    stderr="app.err",
)
```

When resource management is active, omit `num_procs`, `num_nodes`, and GPU counts to use
the worker's assigned resources. Specify them only when the workflow has a validated fixed
geometry. Relevant options include `num_procs`, `num_nodes`, `procs_per_node`, `num_gpus`,
`auto_assign_gpus`, `match_procs_to_gpus`, and `extra_args`.

## Simulation directories and inputs

Concurrent applications writing fixed filenames need isolated directories:

```python
LibeSpecs(
    nworkers=4,
    sim_dirs_make=True,
    ensemble_dir_path="ensemble",
    sim_dir_copy_files=["input.template"],
)
```

Other options include `sim_dir_symlink_files`, `sim_input_dir`, and `use_worker_dirs`.
Inside the simulator, the current working directory is already the simulation directory.
Use relative names there for generated input and output.

If parameters must be written to an input file:

1. Preserve the user's original file.
2. Copy a template into each simulation directory.
3. Render a new application input there using an already-installed templating library or
   a safe application-specific writer.
4. Ensure every marker exactly matches a VOCS input name.

Jinja2 is optional, not mandatory. If used, declare/check that dependency. Do not perform
fragile global string replacement on a scientific input file.

## Parsing and safety

- Define the exact output file/line/key and expected units.
- Verify the task succeeded and the output exists before parsing.
- Reject empty, malformed, non-finite, or stale output.
- Ensure separate tasks cannot read each other's files.
- Avoid `shell=True` and do not concatenate untrusted values into shell commands.
- Match `register_app(app_name=...)` and `submit(app_name=...)` exactly.
- Registered Python scripts may use their interpreter automatically; verify behavior for a
  custom interpreter instead of embedding it into application arguments.
