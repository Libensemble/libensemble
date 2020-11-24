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
  double theta[6];
  double bounds[8][2];
  double Tu, Tl, Hu, Hl, r, Kw;
  double rw, L;
  double numer, denom1, denom2, f;

  //Maybe open outside function
  fh = fopen(filename, "rb");
  fread( theta, sizeof( double ), 6, fh );
  fread( x, sizeof( double ), 3, fh );
  fread( bounds, sizeof( double ), 16, fh );

  // Check bounds
  for (i=0;i < 6;i++) {
    assert(theta[i] >= bounds[i][0]);
    assert(theta[i] <= bounds[i][1]);
  }
  for (i=0;i < 2;i++) {
    assert(x[i] >= bounds[i+6][0]);
    assert(x[i] <= bounds[i+6][1]);
  }

  Tu = theta[0];
  Tl = theta[1];
  Hu = theta[2];
  Hl = theta[3];
  r = theta[4];
  Kw = theta[5];
  rw = x[0];
  L = x[1];

  numer = 2.0 * M_PI * Tu * (Hu - Hl);
  denom1 = 2.0 * L * Tu / (log(r/rw) * pow(rw,2) * Kw);
  denom2 = Tu / Tl;
  f = (numer / (log(r/rw) * (1.0 + denom1 + denom2)));

  // Equivalent to f[xs[:, -1] == 1] = f[xs[:, -1].astype(bool)] ** (1.5)
  if ((int)x[2] == 1) {
    f = pow(f, 1.5);
  }

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
