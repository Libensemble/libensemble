#include <stdio.h>
#include <mpi.h>

/* A C code to burn time without using sleep */
int main(int argc, char **argv)
{
    long long int num, count;
    int ierr, num_procs, rank;
    double sum;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    num = 10000000000;

    /*  for loop terminates when n is less than count */
    sum = 0.0;
    for(count = 1; count <= num; ++count)
    {
        sum += (double)(count/1000000.0);
    }

    /* If sum is printed - task finished */
    printf("Proc %d: Sum = %f\n", rank, sum);

    ierr = MPI_Finalize();
    return 0;
}
