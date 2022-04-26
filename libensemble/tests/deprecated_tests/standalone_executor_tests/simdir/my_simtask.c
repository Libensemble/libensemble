#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int ierr, num_procs, rank, delay, error;

    delay=3;
    error=0;

    if (argc >=3) {
        if (strcmp( argv[1],"sleep") == 0 ) {
            delay = atoi(argv[2]);
        }
    }
    if (argc >=4) {
        if (strcmp( argv[3],"Error") == 0 ) {
            error=1;
        }
    }
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    printf("Hello world sleeping for %d seconds on rank %d of %d\n",delay, rank,num_procs);

    sleep(delay);
    //printf("Woken up on rank %d of %d\n",rank,num_procs);

    if (rank==0) {
        if (error==1) {
            printf("Oh Dear! An non-fatal Error seems to have occurred on rank %d\n",rank);
            sleep(delay);
        }
    }

    ierr = MPI_Finalize();
    return(0);
}
