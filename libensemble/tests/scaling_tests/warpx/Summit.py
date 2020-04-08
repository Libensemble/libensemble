machine_specs = \
    {
        'name': 'summit',
        'sim_app': '/ccs/home/mthevenet/warpx/Bin/main2d.gnu.TPROF.MPI.CUDA.ex',
        'nodes': 1,
        'ranks_per_node': 6,
        'extra_args': '-n 1 -a 1 -g 1 -c 1 --bind=packed:1 --smpiargs="-gpu"',
        'OMP_NUM_THREADS': '1'
    }
