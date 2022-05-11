/* --------------------------------------------------------------------
    Naive Electrostatics Code Example
    This is designed only as an artificial, highly configurable test
    code for a libEnsemble sim func.

    Particles position and charge are initiated by a random stream.
    Particles are replicated on all ranks.
    Each rank computes forces for a subset of particles.
    Particle force arrays are allreduced across ranks.

    Sept 2019:
    Added OpenMP options for CPU and GPU. Toggle in forces_naive function.

    Run executable on N procs:

    mpirun -np N ./forces.x <NUM_PARTICLES> <NUM_TIMESTEPS>

    Author: S Hudson.
-------------------------------------------------------------------- */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#define min(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

// Flags 0 or 1
#define PRINT_HOSTNAME_ALL_PROCS 1
#define PRINT_PARTICLE_DECOMP 0
#define PRINT_ALL_PARTICLES 0
#define CHECK_THREADS 0

static FILE* stat_fp;

// Return elapsed wall clock time from start/end timevals
double elapsed(struct timeval *tv1, struct timeval *tv2) {
    return (double)(tv2->tv_usec - tv1->tv_usec) / 1000000
    + (double)(tv2->tv_sec - tv1->tv_sec);
}

// Print from each thread.
int check_threads(int rank) {
    #if defined(_OPENMP)
    int tid, nthreads;
        #pragma omp parallel private(tid, nthreads)
        {
            nthreads = omp_get_num_threads();
            tid = omp_get_thread_num();
            printf("Rank: %d:   ThreadID: %d    Num threads: %d\n", rank, tid, nthreads);

        }
    #else
        printf("Rank: %d: OpenMP is disabled\n", rank);
    #endif
    return 0;
}

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
//TODO Use parallel RNG - As replicated data can currently do on first rank.
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


// Electrostatics pairwise forces kernel (O(N^2))
// No Eq/Opp - no reduction required (poss adv on fine-grained parallel arch).
double forces_naive(int n, int lower, int upper, particle* parr) {

    int i, j;
    double ret = 0.0;
    double dx, dy, dz, r, force;

    // For GPU/Accelerators
    /*
    #pragma omp target teams distribute parallel for \
                map(to: lower, upper, n) map(tofrom: parr[0:n]) \
                reduction(+: ret) //thread_limit(128) //*/

    // For CPU
    //*
    #pragma omp parallel for default(none) shared(n, lower, upper, parr) \
                             private(i, j, dx, dy, dz, r, force) \
                             reduction(+:ret)  //*/
    for(i=lower; i<upper; i++) {
        for(j=0; j<n; j++){
            if (i==j) {
                continue;
            }
            dx = parr[i].p[0] - parr[j].p[0];
            dy = parr[i].p[1] - parr[j].p[1];
            dz = parr[i].p[2] - parr[j].p[2];
            r = sqrt(dx * dx + dy * dy + dz * dz);

            //force = parr[i].q * parr[j].q / r;
            force = parr[i].q * parr[j].q / (r*r);

            parr[i].f[0] += dx * force;
            parr[i].f[1] += dy * force;
            parr[i].f[2] += dz * force;

            ret += 0.5 * force;
        }
    }
    return ret;
}


// Electrostatics pairwise forces kernel (O(N^2))
// Triangle loop structure (eq/opp)
double forces_eqopp(int n, int lower, int upper, particle* parr) {

    int i, j;
    double ret = 0.0;
    double dx, dy, dz, r, force;

    for(i=lower; i<upper; i++) {
        for(j=i+1; j<n; j++) {

            dx = parr[i].p[0] - parr[j].p[0];
            dy = parr[i].p[1] - parr[j].p[1];
            dz = parr[i].p[2] - parr[j].p[2];
            r = sqrt(dx * dx + dy * dy + dz * dz);

            //ret += parr[i].q * parr[j].q / r;
            force = parr[i].q * parr[j].q / (r*r);

            parr[i].f[0] += dx * force;
            parr[i].f[1] += dy * force;
            parr[i].f[2] += dz * force;

            parr[j].f[0] -= dx * force;
            parr[j].f[1] -= dy * force;
            parr[j].f[2] -= dz * force;

            ret += force;
        }
    }
    return ret;
}


// Currently particles can just move outside initial boundaries.
// May add periodic boundaries
int move_particles(int lower, int upper, particle* parr) {
    int i;
    double dt = 1e-3;
    double dtforce = 0.5*dt;
    double vx, vy, vz;

    // Calculate new positions
    for(i=lower; i<upper; i++) {
        // Without using dim
        // Vel should prob be maintained between steps and modify here.
        vx = dtforce * parr[i].f[0];
        vy = dtforce * parr[i].f[1];
        vz = dtforce * parr[i].f[2];
        parr[i].p[0] += dt * vx;
        parr[i].p[1] += dt * vy;
        parr[i].p[2] += dt * vz;
    }
    return 0;
}

// Print positions of all particles
int print_particles(int n, particle* parr) {
    int i;
    double x, y, z;
    printf("\nPrinting %d particles:\n", n);

    for(i=0; i<n; i++) {
        printf("Point %4d: ", i);

        // Positions
        x = parr[i].p[0];
        y = parr[i].p[1];
        z = parr[i].p[2];
        printf("Pos (%6.3f, %6.3f, %6.3f)", x, y, z);

        // Forces
        x = parr[i].f[0];
        y = parr[i].f[1];
        z = parr[i].f[2];
        printf("   Forces (%7.3f, %7.3f, %7.3f)", x, y, z);
        printf("   Charge: %.2f\n", parr[i].q);
    }
    return 0;
}


int print_step_summary(int step, double total_en,
                       double compute_forces_time,
                       double comms_time) {
    printf("\nStep: %d\n", step);
    printf("Forces kernel returned: %f \n", total_en);
    printf("Forces compute time: %.3f seconds\n", compute_forces_time);
    printf("Forces comms time:   %.3f seconds\n", comms_time);
    return 0;
}


int open_stat_file() {
    char *statfile = "forces.stat";
    stat_fp = fopen(statfile, "w");
    if(stat_fp == NULL) {
        printf("Error opening statfile");
        return 1;
    }
    fflush(stat_fp);
    return 0;
}

int close_stat_file() {
    return fclose(stat_fp);
}

int write_stat_file(double value) {
    fprintf(stat_fp,"%.5f\n", value);
    fflush(stat_fp);
    return 0;
}


int write_stat_file_kill() {
    fprintf(stat_fp,"kill\n");
    fflush(stat_fp);
    return 0;
}

int pack_forces(int n, particle* parr, double forces[][3]) {
    int i, dim;
    for(i=0; i<n; i++) {
        for(dim=0; dim<3; dim++) {
            forces[i][dim] = parr[i].f[dim];
        }
    }
    return 0;
}

int unpack_forces(int n, particle* parr, double forces[][3]) {
    int i, dim;
    for(i=0; i<n; i++) {
        for(dim=0; dim<3; dim++) {
            parr[i].f[dim] = forces[i][dim];
        }
    }
    return 0;
}

int comm_forces(int n, particle* parr) {
    // Note: For square version - a gather would do
    // A reduce works for square or triangle(eq/opp) versions (as forces zeroed each step).
    double forces[n][3];
    pack_forces(n, parr, forces);
    MPI_Allreduce(MPI_IN_PLACE, forces, n*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    unpack_forces(n, parr, forces);
    return 0;
}

int test_badrun(double rate) {
    int bad_run = 0;
    if (get_rand() >= rate) {
        bad_run = 1;
    }
    return bad_run;
}


int main(int argc, char **argv) {

    int num_devices;
    int num_particles = 10; // default no. of particles
    int num_steps = 10; // default no. of timesteps
    int rand_seed = 1; // default seed
    double kill_rate = 0.0; // default proportion of tasks to kill

    int ierr, rank, num_procs, k, m, p_lower, p_upper, local_n;
    int step;
    double compute_forces_time, comms_time, total_time;
    struct timeval tstart, tend;
    struct timeval compute_start, compute_end;
    struct timeval comms_start, comms_end;

    double local_en, total_en;
    double step_survival_rate = pow((1-kill_rate),(1.0/num_steps));
    int badrun = 0;

    if (argc >=2) {
        num_particles = atoi(argv[1]); // No. of particles
    }

    if (argc >=3) {
        num_steps = atoi(argv[2]); // No. of timesteps
    }

    if (argc >=4) {
        rand_seed = atoi(argv[3]); // RNG seed
        seed_rand(rand_seed);
    }

    if (argc >=5) {
        kill_rate = atof(argv[4]); // Proportion of tasks to kill
        step_survival_rate = pow((1-kill_rate),(1.0/num_steps));
    }

    particle* parr = (particle*)malloc(num_particles * sizeof(particle));
    build_system(num_particles, parr);
    //printf("\n");

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("Particles: %d\n", num_particles);
        printf("Timesteps: %d\n", num_steps);
        printf("MPI Ranks: %d\n", num_procs);
        printf("Random seed: %d\n", rand_seed);
    }

     // For multi-gpu node - use one GPU per rank i
     num_devices = 0;
     #if defined(_OPENMP)
     num_devices = omp_get_num_devices();
     if (num_devices > 0) {
         omp_set_default_device(rank % num_devices);
     }
     #endif

    if (PRINT_HOSTNAME_ALL_PROCS) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (num_devices == 0) {
            check_threads(rank);
        }
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        printf("Proc: %d is on node %s\n", rank, processor_name);
    }

    if (CHECK_THREADS) {
        if (num_devices == 0) {
            check_threads(rank);
        }
    }

    k = num_particles / num_procs;
    m = num_particles % num_procs; // Remainder = no. procs with extra particle
    p_lower = rank * k + min(rank, m);
    p_upper = (rank + 1) * k + min(rank + 1, m);
    local_n = p_upper - p_lower;

    if (PRINT_PARTICLE_DECOMP) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Proc: %d has %d particles\n", rank, local_n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    if (rank == 0) {
        open_stat_file();
    }

    gettimeofday(&tstart, NULL);
    for (step=0; step<num_steps; step++) {

        gettimeofday(&compute_start, NULL);

        init_forces(0, num_particles, parr); // Whole array

        local_en = forces_naive(num_particles, p_lower, p_upper, parr);
        //local_en = forces_eqopp(num_particles, p_lower, p_upper, parr);

        gettimeofday(&compute_end, NULL);
        compute_forces_time = elapsed(&compute_start, &compute_end);

        // Note: Will need to add barrier to get pure comms time
        gettimeofday(&comms_start, NULL);

        // Now allreduce forces and update particle positions on first rank

        // Forces array reduction
        comm_forces(num_particles, parr);

        // Scalar reduce energy
        MPI_Allreduce(&local_en, &total_en, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        gettimeofday(&comms_end, NULL);
        comms_time = elapsed(&comms_start, &comms_end);

        // Update positions globally (each rank replicates)
        move_particles(0, num_particles, parr);

        if (!badrun) {
            badrun = test_badrun(step_survival_rate);
        }


        if (rank == 0) {
            print_step_summary(step, total_en, compute_forces_time, comms_time);
            if (badrun) {
                write_stat_file_kill();
            }
            else {
                write_stat_file(total_en);
            }
        }
    }

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tend, NULL);
    total_time = elapsed(&tstart, &tend);

    if (rank == 0) {
        printf("\nFinal total %f after total time of %.3f seconds.", total_en, total_time);
        if (badrun) {
            printf(" Kill flag set.");
        }
        printf("\n");
        fflush(stdout);
    }

    if (PRINT_ALL_PARTICLES) {
        if (rank == 0) {
            print_particles(num_particles, parr);
        }
    }
    if (rank == 0) {
        close_stat_file();
    }
    free(parr); //todo - prob do in teardown routine.
    ierr = MPI_Finalize();
    return 0;
}
