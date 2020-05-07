#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>

/* A parallel C code where each process prints to file at regular intervals */
int main(int argc, char **argv)
{
    int ierr, num_procs, rank, delay, count, num_sleeps;
    time_t start, diff;
    double time_secs;

    delay=1;
    num_sleeps = 16;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    start = time(NULL);;

    printf("Proc %d has started\n", rank);
    fflush(stdout);

    for(count = 0; count < num_sleeps; ++count) {

        sleep(delay);
        /*diff = clock() - start;
        time_secs = (double)diff/CLOCKS_PER_SEC;*/
        time_secs = (double)(time(NULL) - start);


        /* Flush or use stderr */
        printf("Proc %d: Printing after %f seconds\n", rank, time_secs);
        fflush(stdout);
    }

    printf("Proc %d has finished\n", rank);
    fflush(stdout);
    ierr = MPI_Finalize();
    return 0;
}
