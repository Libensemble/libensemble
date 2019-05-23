#!/bin/bash

# GCC
mpicc -O3 -o forces.x forces.c -lm

# Intel
# mpicc -O3 -o forces.x forces.c

# Cray
# cc -O3 -o forces.x forces.c
