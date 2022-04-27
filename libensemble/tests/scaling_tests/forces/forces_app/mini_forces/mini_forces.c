/* --------------------------------------------------------------------
    Non-MPI, Single Step, Electrostatics Code Example

    This is a complete working example to test threading/vectorization
    without MPI or other non-trivial features.

    E.g: gcc build and run on 2 threads on CPU:

    gcc -O3 -fopenmp -o mini_forces.x mini_forces.c -lm
    export OMP_NUM_THREADS=2
    ./mini_forces.x

    E.g: xlc build and run on GPU:

    # First toggle #omp pragma line to target in forces_naive function.
    xlc_r -O3 -qsmp=omp -qoffload -o mini_forces_xlc_gpu.x mini_forces.c
    ./mini_forces.x

    Functionality:
    Particles position and charge are initiated by a random stream.
    Computes forces for all particles. Note: This version uses
    parallel arrays to store the data (SoA).

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
int build_system(int n, double* x, double* y, double* z, double* fx, double* fy, double* fz, double* q) {
    int q_range_low = -10;
    int q_range_high = 10;
    double extent = 10.0;
    int i;

    for(i=0; i<n; i++) {
        x[i] = get_rand()*extent;
        y[i] = get_rand()*extent;
        z[i] = get_rand()*extent;
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
        q[i] = ((q_range_high+1)-q_range_low)*get_rand() + q_range_low;;
    }
    return 0;
}


// Initialize forces to zero for all particles
int init_forces(int lower, int upper, double* fx, double* fy, double* fz) {
    int i;
    for(i=lower; i<upper; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
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
double forces_naive(int n,  double* x, double* y, double* z, double* fx, double* fy, double* fz, double* q) {
    int i, j;
    double ret = 0.0;
    double dx, dy, dz, r, force;
    struct timeval tv1, tv2;

    gettimeofday(&tv1, NULL);

    // For GPU/Accelerators
    /*
    #pragma omp target teams distribute parallel for map(to: n) \
                       map(tofrom: x[:n], y[:n], z[:n], fx[:n], fy[:n], fz[:n], q[:n]) \
                       reduction(+:ret) //thread_limit(128) //*/

    // For CPU
    //*
    #pragma omp parallel for default(none) shared(n, x, y, z, fx, fy, fz, q) \
                             private(i, j, dx, dy, dz, r, force) \
                             reduction(+:ret)  //*/
    for(i=0; i<n; i++) {
        #pragma omp simd private(dx, dy, dz, r, force) reduction(+:fx[i], fy[i], fz[i], ret) // Enable vectorization
        for(j=0; j<n; j++){
            if (i==j) {
                continue;
            }
            dx = x[i] - x[j];
            dy = y[i] - y[j];
            dz = z[i] - z[j];
            r = sqrt(dx * dx + dy * dy + dz * dz);

            force = q[i] * q[j] / (r*r);
            fx[i] += dx * force;
            fy[i] += dy * force;
            fz[i] += dz * force;

            ret += 0.5 * force;
        }
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
    double *pos_x   = malloc(num_particles * sizeof(double));
    double *pos_y   = malloc(num_particles * sizeof(double));
    double *pos_z   = malloc(num_particles * sizeof(double));
    double *force_x = malloc(num_particles * sizeof(double));
    double *force_y = malloc(num_particles * sizeof(double));
    double *force_z = malloc(num_particles * sizeof(double));
    double *charge  = malloc(num_particles * sizeof(double));

    if (CHECK_THREADS) {
        check_threads();
    }

    build_system(num_particles, pos_x, pos_y, pos_z, force_x, force_y, force_z, charge);
    init_forces(0, num_particles, force_x, force_y, force_z);
    local_en = forces_naive(num_particles, pos_x, pos_y, pos_z, force_x, force_y, force_z, charge);
    printf("energy is %f \n", local_en);

    free(pos_x  );
    free(pos_y  );
    free(pos_z  );
    free(force_x);
    free(force_y);
    free(force_z);
    free(charge );
    return 0;
}
