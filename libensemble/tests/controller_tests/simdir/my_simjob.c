#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv) 
{
    int ierr, num_procs, rank, delay;
        
    delay=3;
    if (argc >=3) {
        if (strcmp( argv[1],"sleep") ==0 ) {
            delay = atoi(argv[2]);
        }
    }
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    printf("Hello world sleeping for %d seconds on rank %d of %d\n",delay, rank,num_procs);  
    sleep(delay);
    //printf("Woken up on rank %d of %d\n",rank,num_procs);  
    ierr = MPI_Finalize();   
    return(0);
}
