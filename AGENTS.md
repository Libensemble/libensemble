# Agent Contributor Guidelines

## Architecture

Manager-worker framework: manager allocates points from a generator to workers running simulators. See `libensemble/tests/regression_tests/test_1d_sampling.py` for a minimal example.

## Repository Layout

Core paths relative to `libensemble/`:
- `alloc_funcs/` - Allocation policies.
- `comms/` - Manager-worker communication
- `executors/` - Launching executables
- `gen_classes/` - **Preferred**: gest-api generators
- `gen_funcs/` - Legacy generators
- `resources/` - Compute resource management
- `sim_funcs/` - Simulator functions
- `tests/{functionality,regression,unit}_tests/`
- `ensemble.py` - Primary interface (wraps `libE()`)
- `generators.py` - gest-api base classes
- `history.py` - NumPy structured array for input/output data
- `libE.py` - Entrypoint for libEnsemble, and also legacy top-level interface
- `manager.py` - History & worker coordination
- `specs.py` - `SimSpecs`, `GenSpecs`, `AllocSpecs`, `ExitCriteria`, `LibeSpecs` dataclasses
- `worker.py` - Runs simulators, communicates with manager. Can be configured to run generators as well

## Specifications (Modern Configs)

All configs use **dataclasses** from `specs.py`, not bare dicts (legacy):
- `SimSpecs` - simulator config (`sim_f`/`simulator`, `in`/`inputs`, `out`/`outputs`, `vocs`)
- `GenSpecs` - generator config (`gen_f`/`generator`, `in`/`inputs`, `out`/`outputs`, `persis_in`, `vocs`, `user`)
- `AllocSpecs` - allocation function config (`alloc_f`, `user`)
- `ExitCriteria` - termination conditions (`sim_max`, `wallclock_max`, `stop_val`)
- `LibeSpecs` - runtime config (`comms`, `nworkers`, `gen_on_worker`, `safe_mode`, etc.)

These accept `vocs` from `gest_api.vocs` (not xopt.vocs). The dict-based `sim_specs`/`gen_specs` API still works but is legacy.

## Generators

- **gest-api** (preferred): class inheriting `gest_api.Generator`, implements `suggest(input_dicts)`/`ingest(output_dicts)`, parameterized by `VOCS`. See `libensemble/gen_classes/external/sampling.py`.
- Generators are used for sampling, optimization, calibration, uncertainty quantification, and other simulation-based tasks.
- **Legacy**: plain functions with persistent loops. Use `LibensembleGenerator` to wrap into gest-api.
- History array: NumPy structured array with fields from `sim_specs/gen_specs["out"]` or `vocs` attributes plus reserved metadata fields.
- **Automatic Variable Mapping**: `LibensembleGenerator` maps all VOCS vars to `"x"` field unless `variables_mapping` is provided.
- **Mandatory Input Fields**: `gen_specs["in"]`/`["persis_in"]` must have >=1 field (e.g. `["sim_id"]`) when using `only_persistent_gens` allocator.
- **Default Allocator**: `only_persistent_gens` for gest-api generators.

## Conventions

- Match output fields ↔ input fields (e.g., `SimSpecs.out` ↔ `GenSpecs.in`, and vice-versa).
- Always use the dataclass configs from `specs.py` (`SimSpecs`, `GenSpecs`, etc.) over legacy bare dicts.
- `SimSpecs`/`GenSpecs` accept `vocs` from `gest_api.vocs` (not xopt.vocs).
- Code style: `black` (enforced by pre-commit via `pre-commit`).
- No destructive git commands without explicit request.

## Development

- **pixi** recommended. Enter: `pixi shell -e dev`. One-off: `pixi run -e dev <cmd>`. (Path: `/opt/homebrew/bin/pixi` or `/usr/local/bin/pixi`.)
- Fallback: `pip install -e .` (may need manual dependency installs).
- Pre-commit: `pre-commit` (config in `.pre-commit-config.yaml`, `pyproject.toml`).

## Testing

- Full suite: `python libensemble/tests/run_tests.py`
- Single unit test: `pixi run -e dev pytest path/to/test_file`
- Single regression/functionality test: `pixi run -e dev python path/to/test_file -n 4`
- Check `ensemble.log` and `libE_stats.txt` for run diagnostics.

## Modernizing for libEnsemble 2.0

When updating scripts from legacy patterns:

- **Generators**: Replace `gen_f` with gest-api `Generator` class set via `gen_specs["generator"]`.
- **VOCS**: Parameterize generators with `VOCS` from `gest_api.vocs`.
- **AllocSpecs**: `AllocSpecs` dataclass replaces bare dict. Often removable — `only_persistent_gens` is the default allocator.
- **Generator placement**: Runs on manager (Worker 0) by default. Set `LibeSpecs(gen_on_worker=True)` to run on a dedicated worker.
- **Input fields**: `GenSpecs.inputs`/`persis_in` must contain >=1 field.
- **Simulators**: Use `SimSpecs.simulator` with `(input_dict, **kwargs) -> dict` instead of `sim_f`. libEnsemble auto-wraps via `gest_api_sim`. `inputs`/`outputs` auto-derived from `vocs`.
- **safe_mode**: `LibeSpecs(safe_mode=False)` by default (protected History fields overwritable). Set `True` to guard metadata fields (`gen_worker`, `sim_worker`, `sim_started`, `sim_ended`, etc.).
