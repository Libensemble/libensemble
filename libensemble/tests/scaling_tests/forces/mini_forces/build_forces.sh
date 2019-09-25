#!/bin/bash

# Building with OpenMP for CPU

# GCC for current host (generally not as good at vectorization as intel).
gcc -O3 -fopenmp -march=native -o mini_forces.x mini_forces.c -lm

# Intel (fat binary covering vectorization for hswell/bdwell/skylake/knl)
# icc -O3 -qopenmp -axAVX,CORE-AVX2,CORE-AVX512,MIC-AVX512 -o mini_forces.x mini_forces.c

# Intel with vectorization for current host (no cross-compilation).
# icc -O3 -qopenmp -xhost -o mini_forces.x mini_forces.c

# xl
# xlc_r -O3 -qsimd=auto -qsmp=omp -o mini_forces.x mini_forces.c

# ----------------------------------------------

# Building with OpenMP for target device (e.g. GPU)
# Need to toggle to OpenMP target directive in mini_forces.c.

# xl
# xlc_r -O3 -qsmp=omp -qoffload -o mini_forces.x mini_forces.c

# IRIS node (Intel Gen9 GPU)
# icx -g -fiopenmp -fopenmp-targets=spir64 -o mini_forces.x mini_forces.c
