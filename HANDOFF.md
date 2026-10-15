# Handoff: generator public names, internal transport mapping, and output basenames

## Purpose

This handoff captures the current design discussion and proposed migration for letting users
name simulator variables and outputs naturally (including names such as `x`) while keeping
legacy NumPy-backed algorithms on their established internal fields such as vector `x` and
objective `f`. It also records the `save_output()` basename issue and the current observed
repository state.

## Current repository state

- No generator-mapping migration has been implemented yet.
- The current adapter path is in `libensemble/generators.py` and
  `libensemble/utils/runners.py`; `Ensemble` is the workflow facade, not the right place for
  algorithm-specific mapping logic.
- The current `LibensembleGenerator` auto-creates mappings from internal names such as
  `x`/`f` to VOCS names (`libensemble/generators.py:55-71`). The runner maps fields around
  `suggest_numpy()` and `ingest_numpy()` (`libensemble/utils/runners.py:202-227`). The
  mapping/conversion helpers are in `libensemble/utils/misc.py:314-360,371-481`.
- Built-in `UniformSample` currently subclasses the NumPy adapter and emits one structured
  field per VOCS variable (`libensemble/gen_classes/sampling.py:18-45`). Its returned public
  field named `x` collides with an automatically created internal vector mapping for a
  multi-variable problem also containing a scalar VOCS variable named `x`.
- The collision was reproduced: a two-variable sampling workflow using VOCS fields `x` and
  `y` failed with a structured-array shape mismatch when mapped output was merged with
  public `GenSpecs.outputs`.
- A renamed variable (`x0`, `y`) allowed the built-in sampler workflow to pass locally.
  A custom pure gest-api replacement was attempted during evaluation but did not work
  without additional interface/termination care; do not recommend that as the workaround.
- No changes for the proposed generator architecture are in the tree.

## Mapping contract to target

Public generator interaction must remain name-native. A user calling a generator directly,
including from a notebook, should be able to use records such as:

```python
points = generator.suggest(4)
# [{'x': ..., 'y': ...}, ...]
generator.ingest([{"x": ..., "y": ..., "energy": ...}])
```

Those public records and libEnsemble History should use the VOCS names. An algorithm may
continue to consume/produce a private NumPy representation:

```text
public records / History:  scalar fields x, y, energy
algorithm internal array:  vector field x=[x,y], scalar f=energy
```

The internal representation must be created only at the adapter boundary. It must never
be merged into the public History dtype when names collide.

### Recommended architecture

1. **Define two APIs explicitly.**
   - `suggest()` / `ingest()` are the public gest-api contract and exchange lists of
     dictionaries with VOCS/user names.
   - `suggest_numpy()` / `ingest_numpy()` are adapter implementation methods and exchange
     algorithm-internal structured arrays.
2. **Make mapping a boundary conversion.**
   - Adapter `suggest()`: call the internal NumPy algorithm, then convert internal mapped
     fields (e.g. `x`, `f`) to public variable/objective/observable names and dictionaries.
   - Adapter `ingest(public_records)`: convert records with public names to a public
     structured representation, pack mapped components into the internal array (`x`, `f`),
     then call `ingest_numpy()`.
   - Keep public `GenSpecs.outputs`, manager History, and simulator inputs in VOCS names.
3. **Make the standardized runner call the public methods.**
   - `LibensembleGenRunner` should ask the generator for public records via `suggest()` and
     convert those records directly to the History output dtype.
   - Convert returned History fields to public dictionaries and call `ingest()`.
   - Do not separately map arrays into algorithm-private fields in the runner; that creates
     a second conversion path and exposes internal names to History.
4. **Separate native gest-api generators from adapters.**
   - Simple generators such as uniform and Latin-hypercube sampling do not need legacy
     NumPy `x`/`f` representations. Prefer implementing them as pure gest-api generators
     that naturally operate on variable-name dictionaries.
   - Keep `LibensembleGenerator` for algorithms that genuinely require the legacy NumPy
     interface, including APOSMM and other applicable adapters.
5. **Audit all adapter subclasses before changing semantics.**
   - Identify what each subclass currently returns from `suggest_numpy()` and expects in
     `ingest_numpy()`; these conventions are not necessarily uniform today.
   - Preserve internal algorithm contracts while normalizing only the adapter boundary.
   - Handle `H0`, initial sampling, constants, objectives, observables, constraints,
     `_id`/`sim_id`, optional mappings, vectors, and `suggest_updates()` explicitly.
6. **Avoid fragile field aliasing.**
   - A private transport field could be generated to avoid collisions, but a separate
     internal array prevents History contamination more simply.
   - Do not assume a user mapping can make a single structured-array field both scalar and
     vector. Do not fix the issue by silently renaming user VOCS fields.

## Migration plan

### Phase 1: Characterize existing behavior

- Inventory all `LibensembleGenerator` subclasses and their `suggest_numpy()` /
  `ingest_numpy()` field names and shapes.
- Record the current meaning/direction of `variables_mapping` for each class.
- Add focused tests for mapping round trips, including:
  - public variables named `x` and `y`, with an internal vector `x`;
  - variable and objective names `x`/`f`;
  - two or more variables/objectives;
  - explicit mappings and unmapped extra fields;
  - scalar/vector/categorical fields where supported;
  - generator initialization from `H0` and `initial_sample_method`;
  - direct notebook-style `suggest()` then `ingest()` calls.

### Phase 2: Specify and implement adapter conversions

- Document the public-versus-internal contract in `LibensembleGenerator`.
- Implement conversion helpers in one place (adapter code), not independently in each
  runner branch.
- Change `LibensembleGenRunner` to use the public dict API for suggestions and ingestion.
- Ensure `GenSpecs` derives public fields from VOCS and only those public fields enter
  History.
- Retain specialized APIs such as APOSMM's local minima and internal run state without
  exposing algorithm-private `x`/`f` arrays as replacement user fields.

### Phase 3: Normalize native generators

- Convert native sampling generators to pure gest-api classes where practical.
- Preserve their seed, batch, dtype, constants, and `suggest`/`ingest` behavior.
- Confirm generator exhaustion and exact `sim_max` termination behavior.
- Consider compatibility impact if a public class currently inherits `LibensembleGenerator`
  or exposes NumPy-specific methods.

### Phase 4: Integration and compatibility tests

- Run the direct generator round-trip tests and full Ensemble workflows.
- Cover local spawned workers (macOS/Windows behavior), manager-side generator threads,
  `H0`, initial sample generation, partial/async returns, and generator exhaustion.
- Ensure the internal algorithm receives byte-for-byte/shape-equivalent fields where its
  established contract requires them.
- Run relevant APOSMM, sampling, generator, and Ensemble unit/regression tests.
- Add a regression test reproducing the original `x`/`y` collision and assert public History
  contains separate scalar `x` and `y` fields with correct values.

### Phase 5: Documentation and migration notes

- Document that `suggest()`/`ingest()` are public VOCS-name APIs and NumPy methods are
  adapter internals.
- Explain that internal `variables_mapping` does not rename the user's public records.
- Update examples and any generator API guidance to match the final behavior.

## Open design questions

1. Should `suggest_numpy()` / `ingest_numpy()` be formally private (e.g. `_suggest_numpy` /
   `_ingest_numpy`) or remain public for backward compatibility?
2. Do any users rely on `LibensembleGenerator.suggest()` returning mapped internal fields
   rather than VOCS names? Audit behavior and release compatibility before changing it.
3. Should native samplers be migrated to pure gest-api implementations in the same change,
   or should they first use the corrected adapter boundary?
4. What is the intended public meaning of explicit `variables_mapping` for a generator
   instantiated and used outside an Ensemble?
5. How should constants be added for generators that do not emit them, while retaining
   direct `suggest()` usability?
6. Which metadata is legitimate to expose in public generator records (`sim_id`, `_id`,
   `local_min`, etc.) versus strictly internal to libEnsemble?
7. Should mapping be generalized to variable/objective groups only, or support arbitrary
   internal algorithm fields such as gradients, residuals, constraints, and auxiliary state?

## Separate issue: output basename normalization

The basename bug was that `_get_shortname()` used `script_name.split("test_", 1).pop()`,
which removed everything before the first occurrence of `test_` anywhere in the filename.
For example, `sampling_retest_renamed.py` unexpectedly became `renamed.npy`.

The intended behavior is to remove only a leading `test_` used by test calling-script
names. The implementation currently visible in `libensemble/tools/tools.py:66-68` is:

```python
script_name = os.path.splitext(os.path.basename(basename))[0]
return script_name.removeprefix("test_")
```

The current visible `Ensemble.save_output()` implementation at
`libensemble/ensemble.py:489-511` returns the resulting History filename on the manager
(and `None` elsewhere) and documents this normalization. Regression tests currently visible
in `libensemble/tests/unit_tests/test_ensemble.py:26-54` cover both an embedded `test_`
(`sampling_retest_renamed.py`) and a leading test prefix (`test_sampling.py`).

In the preceding work session, the targeted Ensemble unit test passed:

```text
28 passed in 3.29s
```

At handoff time, the relevant files showed no uncommitted changes in `git status`, so these
changes appeared to be part of the current checkout state. Re-run the tests and inspect
`git log`/`git status` before assuming whether they are committed or pre-existing in a later
checkout.

## Files to start with

- `libensemble/generators.py:19-117` — adapter, mappings, public wrapper methods.
- `libensemble/utils/runners.py:84-122,202-227` — standardized and adapter runner paths.
- `libensemble/utils/misc.py:308-360,371-514` — record/array conversion helpers.
- `libensemble/specs.py:317-374` — VOCS-derived fields and generator outputs.
- `libensemble/gen_classes/sampling.py:18-95` — adapter-based uniform/LHS samplers.
- `libensemble/gen_classes/aposmm.py` — algorithm with internal vector mapping requirements.
- `libensemble/tests/unit_tests/test_ensemble.py:26-54` — output basename regressions.

## Suggested first implementation task

Start with a focused design/test PR rather than a broad refactor:

1. Add a failing regression test for a `UniformSample` workflow with public variables `x`
   and `y`, asserting History contains scalar `x` and `y`.
2. Add direct adapter public API tests for `suggest()` and `ingest()` using those names.
3. Trace current class-specific NumPy contracts and select one explicit internal/public
   conversion boundary.
4. Implement the smallest change that keeps APOSMM's internal `x` vector untouched while
   keeping user-facing History and notebook records VOCS-named.
5. Run focused generator/runner tests before expanding migration to other adapters.
