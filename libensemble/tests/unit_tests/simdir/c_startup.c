#include <sys/time.h>
#include <stdio.h>

int main (void)
{
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    printf ("%f\n",
         (double) (tv.tv_usec) / 1000000 +
         (double) (tv.tv_sec));
}
