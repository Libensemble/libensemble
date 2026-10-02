# HPC resources and launch modes

The batch scheduler, libEnsemble manager/worker transport, and simulation application
launcher are separate layers. Do not conflate them.

## Default cluster pattern

1. A Slurm/PBS/LSF script requests an allocation.
2. The calling script starts once, commonly with plain `python workflow.py` and local
   workers configured by `LibeSpecs(nworkers=N)`.
3. `MPIExecutor` launches each simulation inside resources assigned to its worker.

Do not launch the calling script with `mpirun`, `srun`, or `mpiexec` merely because the
simulation application uses MPI. Distributed `mpi4py` manager/workers are a distinct mode
and must be explicitly requested and configured.

## Resource sets

```python
libE_specs = LibeSpecs(
    comms="local",
    nworkers=4,
    num_resource_sets=4,
    safe_mode=True,
    sim_dirs_make=True,
)
```

A resource set is libEnsemble's schedulable partition of detected resources. If
`num_resource_sets` is omitted, resources are normally divided by workers. Standardized
generators run on the manager by default, so do not subtract one worker/resource set unless
`gen_on_worker=True`.

Prefer scheduler/platform auto-detection. Use an override only from known machine facts:

```python
resource_info={
    "cores_on_node": (physical_cores, logical_cores),
    "gpus_on_node": gpu_count,
}
```

Never invent these values. `disable_resource_manager=True` disables resource detection and
assignment and should not be boilerplate.

## MPIExecutor submission

Normally let assigned resources determine geometry:

```python
task = executor.submit(app_name="solver", app_args="--case input.dat")
```

For a validated fixed request:

```python
task = executor.submit(
    app_name="solver",
    num_procs=8,
    num_nodes=2,
    procs_per_node=4,
    num_gpus=2,
    auto_assign_gpus=True,
)
```

The request must fit resources allocated to that simulation. Avoid hard-coded node names or
machinefiles unless the deployment specifically requires them.

## Variable-size tasks

Resource-aware standardized generator classes may emit reserved resource request fields
such as `num_procs`, `num_gpus`, or `resource_sets`. Prefer an existing tested class for
this behavior. A custom pure gest-api generator must deliberately define and map these
fields; do not add them as ordinary VOCS objectives or overwrite unrelated protected
History metadata.

## Platform and runner settings

`MPIExecutor()` can auto-detect an MPI runner. A known platform may instead provide
`platform_specs`, or `MPIExecutor(custom_info={"mpi_runner": "srun"})` may select a runner.
Valid launch behavior is system-specific; confirm it with site documentation.

Do not guess:

- scheduler account, queue/partition, walltime, node count;
- cores, hardware threads, GPUs, tiles, or affinity;
- MPI implementation/runner and runner-specific flags;
- whether manager/worker processes need dedicated nodes;
- environment modules or activation commands.

## Scheduler script guidance

Generate a scheduler script only when requested and machine facts are supplied. Request
nodes/resources in scheduler directives, then put per-simulation process/GPU geometry in
libEnsemble resource requests or `MPIExecutor.submit()`. On systems with nested job steps,
overly restrictive per-task scheduler directives can prevent child launches.

If the user requests CLI-selectable worker counts, construct `Ensemble(parse_args=True)`
and show the matching invocation. Otherwise put `nworkers` in `LibeSpecs` and invoke the
script without `-n` or `--comms`.

## GPU checks

- Confirm whether one GPU maps to one rank or several ranks.
- Confirm whether the application uses visible-device environment variables or launcher
  GPU flags.
- Use `auto_assign_gpus`/`match_procs_to_gpus` only with compatible runner behavior.
- Ensure aggregate concurrent requests fit the allocation.
- Keep CPU-only and GPU task requirements explicit in heterogeneous workflows.
