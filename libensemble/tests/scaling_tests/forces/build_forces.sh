#!/bin/bash

# GCC
mpicc -o forces.x forces.c -lm

# Intel
# mpicc -o forces.x forces.c

# Cray
# cc -O3 -o forces.x forces.c
