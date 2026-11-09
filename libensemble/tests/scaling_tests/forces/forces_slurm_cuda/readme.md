## Slurm + CUDA (gpu_sharing_factor, spread)

Variable-GPU forces example for Slurm systems (Perlmutter) on **real NVIDIA GPUs** - no mocking.

`gpu_sharing_factor=2` (4 GPUs/node → 8 shares/node). Generator cycles `0.5/1.0/1.5/2.0` → 1-4 shares with `spread` packing; tests cross-node splitting. `SLURM_NODELIST` and NVIDIA GPUs auto-detected.

### Build

In `../forces_app`:

```bash
# Perlmutter (Cray EX, A100)
cc -DGPU -O3 -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c
# Generic NVIDIA
mpicc -DGPU -O3 -fopenmp -mp=gpu -o forces.x forces.c
```

Requires `forces.x` with `-DGPU` and real NVIDIA GPUs.

### Run

```bash
sbatch submit_slurm.sh   # --nodes 1-2, srun -n N python run_libe_forces.py
```

Manager asserts `shares==ceil(req*sf)`∈1-4, cycle coverage, `total_gpu_rsets>0`, `gpu_rsets_per_node`, `even_groups`/`ngroups`; worker asserts `num_rsets`, `local_node_count` splitting, `matching_slots`.
