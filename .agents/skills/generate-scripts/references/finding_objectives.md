# Finding Objective Fields
How to find objective field names in results files.

## VOCS scripts

The objective field name is defined in the VOCS object:
```python
vocs = VOCS(
    variables={"x0": [-2, 2], "x1": [-1, 1]},
    objectives={"f": "MINIMIZE"},
)
```

The key in `objectives` (e.g. `"f"`) is the objective field name in the results.

## Classic scripts

The objective field name is defined in `sim_specs` outputs:
```python
sim_specs = SimSpecs(
    ...
    outputs=[("f", float)],  # "f" is the objective field name
)
```

The field name in `outputs` (e.g. `"f"`) matches the field name in the `.npy` results file.

## Common patterns
- Single objective: `{"f": "MINIMIZE"}` (VOCS) or `outputs=[("f", float)]` (classic)
- Multiple outputs: `"f"` is typically the objective — the scalar float used by the generator
- The objective field name in the VOCS definition or sim_specs outputs matches the field in the results
