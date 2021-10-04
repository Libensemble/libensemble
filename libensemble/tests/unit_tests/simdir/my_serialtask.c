#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    int usec_delay, error;
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

    printf("Hello world sleeping for %f seconds\n",fdelay);
    usec_delay = (int)(fdelay*1e6);
    usleep(usec_delay);

    if (error==1) {
        printf("Oh Dear! An non-fatal Error seems to have occurred\n");
        fflush(stdout);
        usleep(usec_delay);
    }

    return(0);
}
