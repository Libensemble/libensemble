# Results Metadata
How to interpret libEnsemble history array fields and filter for completed simulations.

## History array (H)

The `.npy` output file contains the history array H with both user-defined fields and
metadata fields added by libEnsemble.

## Key metadata fields

- `sim_ended`: True if the simulation completed. Only rows with `sim_ended == True` have valid results.
- `sim_started`: True if the simulation was dispatched to a worker.
- `returned`: True if results were returned to the manager.
- `sim_id`: Unique simulation ID (0-indexed).
- `gen_informed`: True if the generator has been informed of this result.

## Filtering for valid results

When analyzing results (e.g., finding the minimum objective value), always filter for
completed simulations:

```python
H = np.load("results.npy")
done = H[H["sim_ended"]]  # Only completed simulations
```

Rows where `sim_ended` is False may have default/zero values that are not real results.
This is common for the last few rows when the simulation budget is exhausted — they were
allocated by the generator but never evaluated.

## Reporting results

After a successful run, report any minima found in the results. See the generator-specific
guide for which fields indicate identified minima.

## Common pitfall

If the minimum objective value is exactly 0.0, check whether those rows have
`sim_ended == True`. Unevaluated rows often have fields initialized to zero.
This is common for the last few rows when the simulation budget is exhausted — they were
allocated by the generator but never evaluated.

