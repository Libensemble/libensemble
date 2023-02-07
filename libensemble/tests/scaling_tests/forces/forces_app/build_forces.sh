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

# xl
# xlc_r -WF,-DGPU -O3 -qsmp=omp -qoffload -o forces.x forces.c

# Nvidia (nvc) compiler with mpicc and on Cray system with target (Perlmutter)
# mpicc -DGPU -O3 -fopenmp -mp=gpu -o forces.x forces.c
# cc -DGPU -O3 -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c

# Spock/Crusher (AMD ROCm compiler)
# cc -DGPU -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64 -fopenmp -O3 -o forces.x forces.c

# Intel oneAPI (Clang based) Compiler (JIT compiled for device)
# mpiicx -DGPU -O3 -fiopenmp -fopenmp-targets=spir64 -o forces.x forces.c
