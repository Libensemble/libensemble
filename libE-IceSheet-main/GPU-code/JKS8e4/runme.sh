#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

# compile the code
damp=0.98
relaxation=0.99

# compile the code
nvcc -arch=sm_70 -O3 -lineinfo ssa_fem_pt.cu -Ddmp=$damp -Drela=$relaxation 

#To run the code and generate NSIGHT Systems report
nsys profile --stats=true  ./a.out   





