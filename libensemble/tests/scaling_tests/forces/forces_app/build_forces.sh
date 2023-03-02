#!/bin/bash

# Building flat MPI

# GCC
mpicc -O3 -o forces.x forces.c -lm

# Intel
# mpiicc -O3 -o forces.x forces.c

# Cray
# cc -O3 -o forces.x forces.c

# ----------------------------------------------

# Building with OpenMP for CPU

# GCC
# mpicc -O3 -fopenmp -o forces.x forces.c -lm

# Intel
# mpiicc -O3 -qopenmp -o forces.x forces.c

# Cray / Intel (for CCE OpenMP is recognized by default)
# cc -O3 -qopenmp -o forces.x forces.c

# xl
# xlc_r -O3 -qsmp=omp -o forces.x forces.c

# ----------------------------------------------

# Building with OpenMP for target device (e.g. GPU)
# Need to toggle to OpenMP target directive in forces.c.

# xl (plain and using mpicc on Summit)
# xlc_r -DGPU -O3 -qsmp=omp -qoffload -o forces.x forces.c
# mpicc -DGPU -O3 -qsmp=omp -qoffload -o forces.x forces.c

# Summit with gcc (Need up to offload capable gcc: module load gcc/12.1.0) - slower than xlc
# mpicc -DGPU -Ofast -fopenmp -Wl,-rpath=/sw/summit/gcc/12.1.0-0/lib64 -lm -foffload=nvptx-none forces.c -o forces.x

# Nvidia (nvc) compiler with mpicc and on Cray system with target (Perlmutter)
# mpicc -O3 -fopenmp -mp=gpu -o forces.x forces.c
# cc -O3 -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c

# Spock/Crusher (AMD ROCm compiler)
# cc -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64 -fopenmp -O3 -o forces.x forces.c

# Intel oneAPI (Clang based) Compiler (JIT compiled for device)
# mpiicx -O3 -fiopenmp -fopenmp-targets=spir64 -o forces.x forces.c
