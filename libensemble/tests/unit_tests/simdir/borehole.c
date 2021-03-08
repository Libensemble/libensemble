/*
 * Compute Single borehole evaluation with optional delay
 * For testing subprocessing of evaluation and possible kill.
 * Author: S Hudson.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

double borehole_func(char *filename){
  FILE* fh;
  int i,j;
  double x[3];
  double theta[4];
  double Hu, Ld_Kw, Treff, powparam, rw, Hl, L;
  double numer, denom1, denom2, f;

  //Maybe open outside function
  fh = fopen(filename, "rb");
  fread( theta, sizeof( double ), 4, fh );
  fread( x, sizeof( double ), 3, fh );

  Hu = theta[0];
  Ld_Kw = theta[1];
  Treff = theta[2];
  powparam = theta[3];
  rw = x[0];
  Hl = x[1];

  numer = 2.0 * M_PI * (Hu - Hl);
  denom1 = 2.0 * Ld_Kw / pow(rw,2);
  denom2 = Treff;
  f = numer / (denom1 + denom2) * exp(powparam * rw);

  fclose(fh);
  return f;
}

int main(int argc, char **argv){

  char* filename;
  double delay;
  double f;

  if (argc >=2) {
    filename = argv[1]; // input file
  }
  else {
    fprintf(stderr,"No input file supplied");
    exit(EXIT_FAILURE);
  }

  if (argc >=3) {
    delay = atof(argv[2]); // delay in seconds
    // fprintf(stderr, "delay is %f\n",delay);
  }

  sleep(delay); // Simulate longer function
  f = borehole_func(filename);
  printf("%.*e\n",15, f); // Print result to standard out.
  fflush(stdout);

  return 0;
}
