/* --------------------------------------------------------------------
    Non-MPI, Single Step, Electrostatics Code Example

    This is a complete working example to test threading/vectorization
    without MPI or other non-trivial features.

    E.g: gcc build and run on 2 threads on CPU:

    gcc -O3 -fopenmp -march=native -o mini_forces_AoS.x mini_forces_AoS.c -lm
    export OMP_NUM_THREADS=2
    ./mini_forces_AoS.x

    E.g: xlc build and run on GPU:

    # First toggle #omp pragma line to target in forces_naive function.
    xlc_r -O3 -qsmp=omp -qoffload -o mini_forces_AoS_xlc_gpu.x mini_forces_AoS.c
    ./mini_forces_AoS.x

    Functionality:
    Particles position and charge are initiated by a random stream.
    Computes forces for all particles. Note: This version uses
    an array of structures to store the data (AoS). The forces loop
    should vectorize but will have overhead of gathers.

    OpenMP options for CPU and GPU. Toggle in forces_naive function.

    Author: S Hudson.
-------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define CHECK_THREADS 1
#define NUM_PARTICLES 6000

typedef struct particle {
    double p[3]; // Particle position
    double f[3]; // Particle force
    double q;    // Particle charge
}__attribute__((__packed__)) particle;

// Seed RNG
int seed_rand(int seed) {
    srand(seed);
    return 0;
}

// Return a random number from a persistent stream
double get_rand() {
    double randnum;
    randnum = (double)rand()/(double)(RAND_MAX + 1.0); //[0, 1)
    return randnum;
}


// Particles start at random locations in 10x10x10 cube
int build_system(int n, particle* parr) {
    int q_range_low = -10;
    int q_range_high = 10;
    double extent = 10.0;
    int i, dim;

    for(i=0; i<n; i++) {
        for(dim=0; dim<3; dim++) {
            parr[i].p[dim] = get_rand()*extent;
            parr[i].f[dim] = 0.0;
        }
        parr[i].q = ((q_range_high+1)-q_range_low)*get_rand() + q_range_low;
    }
    return 0;
}


// Initialize forces to zero for all particles
int init_forces(int lower, int upper, particle* parr) {
    int i, dim;
    for(i=lower; i<upper; i++) {
        for(dim=0; dim<3; dim++) {
            parr[i].f[dim] = 0.0;
        }
    }
    return 0;
}


// Print from each thread.
int check_threads() {
    int tid, nthreads;

    #pragma omp parallel private(tid, nthreads)
    {
        #if defined(_OPENMP)
            nthreads = omp_get_num_threads();
            tid = omp_get_thread_num();
            printf("ThreadID: %d    Num threads: %d\n", tid, nthreads);
        #else
            printf("OpenMP is disabled\n");
        #endif
    }
    return 0;
}


// Electrostatics pairwise forces kernel (O(N^2))
double forces_naive(int n,  particle* parr) {
    int i, j;
    double ret = 0.0;
    double dx, dy, dz, r, force;
    double fx, fy, fz;
    struct timeval tv1, tv2;

    gettimeofday(&tv1, NULL);

    // For GPU/Accelerators
    /*
    #pragma omp target teams distribute parallel for map(to: n) \
                       map(tofrom: parr[0:n]) \
                       reduction(+:ret) //thread_limit(128) //*/

    // For CPU
    //*
    #pragma omp parallel for default(none) shared(n, parr) \
                             private(i, j, dx, dy, dz, r, force, fx, fy, fz) \
                             reduction(+:ret)  //*/
    for(i=0; i<n; i++) {
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
        #pragma omp simd private(dx, dy, dz, r, force) reduction(+:fx, fy, fz, ret) // Enable vectorization
        for(j=0; j<n; j++){
            if (i==j) {
                continue;
            }
            dx = parr[i].p[0] - parr[j].p[0];
            dy = parr[i].p[1] - parr[j].p[1];
            dz = parr[i].p[2] - parr[j].p[2];
            r = sqrt(dx * dx + dy * dy + dz * dz);

            force = parr[i].q * parr[j].q / (r*r);

            fx += dx * force;
            fy += dy * force;
            fz += dz * force;

            ret += 0.5 * force;
        }
        parr[i].f[0] += fx;
        parr[i].f[1] += fy;
        parr[i].f[2] += fz;
    }

    gettimeofday(&tv2, NULL);
    printf ("Time taken for loop (Wallclock) = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));

    return ret;
}


int main(int argc, char **argv) {

    int num_particles = NUM_PARTICLES;
    double local_en;
    particle* parr = malloc(num_particles * sizeof(particle));


    if (CHECK_THREADS) {
        check_threads();
    }

    build_system(num_particles, parr);
    init_forces(0, num_particles, parr); // Whole array
    local_en = forces_naive(num_particles, parr);
    printf("energy is %f \n", local_en);
    free(parr);
    return 0;
}
