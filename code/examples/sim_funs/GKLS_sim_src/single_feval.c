/*****************************************************************************/
/*   An example of the use of the                                            */
/*    GKLS-Generator of Classes of ND (non-differentiable),                  */
/*                                 D  (continuously differentiable), and     */
/*                                 D2 (twice continuously differentiable)    */
/*    Test Functions for Global Optimization                                 */
/*                                                                           */
/*   In this example, first, a class (with default parameters) of            */
/*   100 D-type test functions is generated and the files with the function  */
/*   information and with the functions points (in the two-dimensional case) */
/*   are created;                                                            */
/*   second, a file with information about gradients and Hessian of specific */
/*   D- and D2-type test functions at a given point is provided.             */
/*                                                                           */
/*   Authors:                                                                */
/*                                                                           */
/*   M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                        */
/*                                                                           */
/*   (C) 2002-2005                                                           */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h> /* for getopt */
// #include <malloc.h>
#include "gkls.h"
#include <math.h>
#include "rnd_gen.h"

void print_error_msg  (int);

/* Print an error message */
void print_error_msg (int error_code)
{
  switch( error_code )
  {
    case GKLS_OK:
		printf("\nGKLS_OK: There is no error.");
		break;
    case GKLS_DIM_ERROR:
		printf("\nGKLS_DIM_ERROR: The problem dimension is out of the valid range [1,%u].",
			  (unsigned int)NUM_RND);
		break;
    case GKLS_NUM_MINIMA_ERROR:
		printf("\nGKLS_NUM_MINIMA_ERROR: The number of local minima must be greater than 1.");
		break;
    case GKLS_FUNC_NUMBER_ERROR:
		printf("\nGKLS_FUNC_NUMBER_ERROR: The number of the test function to be generated is out of the range [1,100].");
		break;
    case GKLS_BOUNDARY_ERROR:
		printf("\nGKLS_BOUNDARY_ERROR: The admissible region boundary vectors are not defined or ill-defined.");
		break;
    case GKLS_GLOBAL_MIN_VALUE_ERROR:
		printf("\nGKLS_GLOBAL_MIN_VALUE_ERROR: The global minimum value must be greater than %f.",
			  (double)GKLS_PARABOLOID_MIN);
		break;
    case GKLS_GLOBAL_DIST_ERROR:
		printf("\nGKLS_GLOBAL_DIST_ERROR: The distance from the paraboloid vertex to the global minimizer is too great.");
		break;
    case GKLS_GLOBAL_RADIUS_ERROR:
		printf("\nGKLS_GLOBAL_RADIUS_ERROR: The radius of the attraction region of the global minimizer is too high.");
		break;
    case GKLS_MEMORY_ERROR:
		printf("\nGKLS_MEMORY_ERROR: There is not enough memory to allocate.");
		break;
    case GKLS_DERIV_EVAL_ERROR:
		printf("\nGKLS_DERIV_EVAL_ERROR: An error occurs during derivative evaluation.");
		break;
    case GKLS_FLOATING_POINT_ERROR:
    default :
		printf("\nUnknown error.");
  }
  printf("\n");
} /* print_error_msg() */


int main(int argc, char **argv)
{
  unsigned int i;    /* cycle parameters     */
  int error_code;    /* error codes variable */
  int func_num;      /* test function number within a class     */
  double *xx;        /* for evaluation of the test function */
  double z;          /* test function value  */
  FILE *GKLS_info;   /* file pointers */
  FILE *x_file_name; /* file pointers */
  char filename[12]; /* name of files */
  char xinfile[12]; /* name of files */
  int rank;          /* rank doing the function evaluation*/


  /* Process the command-line arguments */
  int c;
  while (1)
    {
      static struct option long_options[] =
        {
          /* These options don’t set a flag.
             We distinguish them by their indices. */
          {"dimension",    required_argument, 0, 'd'},
          {"numopt",       required_argument, 0, 'n'},
          {"funcnum",      required_argument, 0, 'f'},
          {"rank",         required_argument, 0, 'r'},
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "d:n:f:r:", long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
        break;

      switch (c)
        {
        case 'd':
          /*printf ("option -d with value `%s'\n", optarg);*/
          GKLS_dim = atoi(optarg);
          break;

        case 'n':
          /*printf ("option -n with value `%s'\n", optarg);*/
          GKLS_num_minima = atoi(optarg);
          break;

        case 'f':
          /*printf ("option -f with value `%s'\n", optarg);*/
          func_num = atoi(optarg);
          break;

        case 'r':
          /*printf ("option -r with value `%s'\n", optarg);*/
          rank = atoi(optarg);
          break;

        case '?':
          /* getopt_long already printed an error message. */
          break;

        default:
          abort ();
        }
    }

  /*[> Print any remaining command line arguments (not options). <]*/
  /*if (optind < argc)*/
  /*  {*/
  /*    printf ("non-option ARGV-elements: ");*/
  /*    while (optind < argc)*/
  /*      printf ("%s ", argv[optind++]);*/
  /*    putchar ('\n');*/
  /*  }*/

  /*GKLS_dim = 2;                                             */
  /*GKLS_num_minima = 10;                                     */
  if ((error_code = GKLS_domain_alloc()) != GKLS_OK)        
    return error_code;                                     
  for (i=0; i<GKLS_dim; i++) 
  {
     GKLS_domain_left[i]  = 0.0;
     GKLS_domain_right[i] = 1.0; 
  }
  GKLS_global_dist = sqrt(GKLS_dim)/2;                               
  /*GKLS_global_dist = 0.499;                               */
  GKLS_global_radius = 0.5*GKLS_global_dist;                
  GKLS_global_value = GKLS_GLOBAL_MIN_VALUE;                
  if ((error_code = GKLS_parameters_check()) != GKLS_OK)    
    return error_code;                                     

  /* Allocate memory for the vector xx of feasible point */
  if ( (xx=(double *)malloc((size_t)GKLS_dim*sizeof(double))) == NULL)
    return (-1);

  /* Generate the class of 100 D-type functions */
  /*func_num=1;*/

  if((error_code=GKLS_arg_generate (func_num)) != GKLS_OK) {
    print_error_msg(error_code);
    return error_code;
  }

  /* Read in x-value */
  /*xx[0]=0.4; xx[1]=0.5;*/
  sprintf(xinfile, "x%4.4d.in",rank);

  x_file_name = fopen(xinfile,"r");
  for (i=0; i<GKLS_dim; i++)
  {
    fscanf(x_file_name,"%lf", &xx[i]);
  }

  z=GKLS_D_func(xx);

  /* File of the test function information */
  /*sprintf(filename,"fout%03d.txt",func_num);*/
  sprintf(filename,"f%4.4d.out",rank);
  if ((GKLS_info=fopen(filename,"wt")) == NULL) return (-1);
  fprintf(GKLS_info,"%20.18f \n",z);
  fclose(GKLS_info);

  /* Deallocate memory */
  GKLS_free();
  free(xx);
  GKLS_domain_free();

  return 0;

} /* example.c */
