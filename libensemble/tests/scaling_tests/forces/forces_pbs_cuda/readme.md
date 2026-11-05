## PBSpro + CUDA (gpu_sharing_factor)

Fractional-GPU forces example for PBSpro systems (Aurora/Polaris) on **real NVIDIA GPUs** - no mocking.

`gpu_sharing_factor=2` splits each NVIDIA GPU into 2 logical shares (4 GPUs/node → 8 shares/node). Generator cycles `0.5/1.0/1.5/2.0` GPUs → 1-4 shares; `pack` packing co-locates fractional tasks. PBS nodelist (`PBS_NODEFILE`) and GPUs are auto-detected.

### Build

In `../forces_app`:

```bash
# Aurora (Intel oneAPI)
mpicc -DGPU -O3 -fiopenmp -fopenmp-targets=spir64 -o forces.x forces.c
# Polaris (NVHPC)
mpicc -DGPU -O3 -fopenmp -mp=gpu -o forces.x forces.c
```

Requires `forces.x` with `-DGPU` and real NVIDIA GPUs in allocation.

### Run

```bash
qsub submit_pbs.sh        # select=1-2, n=5-9, mpirun -np N python run_libe_forces.py
```

Assertions (in sim + manager) verify `num_rsets==ceil(req*sf)`, `gpu_id` valid, `local_node_count` splitting (≤ shares/node → 1 node, else ceil), `matching_slots`, and `rsets_free`/`gpu_rsets` invariants.
