#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int ierr, num_procs, rank, usec_delay, error;
    double fdelay;

    fdelay=3.0;
    error=0;

    if (argc >=3) {
        if (strcmp( argv[1],"sleep") == 0 ) {
            fdelay = atof(argv[2]);
        }
    }
    if (argc >=4) {
        if (strcmp( argv[3],"Error") == 0 ) {
            error=1;
        }
    }
    if (argc >=4) {
        if (strcmp( argv[3],"Fail") == 0 ) {
            return(1);
        }
    }
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    printf("Hello world sleeping for %f seconds on rank %d of %d\n",fdelay, rank,num_procs);

    usec_delay = (int)(fdelay*1e6);
    usleep(usec_delay);
    //printf("Woken up on rank %d of %d\n",rank,num_procs);

    if (rank==0) {
        if (error==1) {
            printf("Oh Dear! An non-fatal Error seems to have occurred on rank %d\n",rank);
            fflush(stdout);
            usleep(usec_delay);
        }
    }

    ierr = MPI_Finalize();
    return(0);
}
