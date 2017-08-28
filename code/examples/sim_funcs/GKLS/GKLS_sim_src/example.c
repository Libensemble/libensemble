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


int main()
{
unsigned int i, j, i2, j2; /* cycle parameters     */
int error_code;    /* error codes variable */
int func_num;      /* test function number within a class     */
double *xx, dx1, dx2; /* for evaluation of the test function */
                      /* on a grid with steps dx1, dx2       */
double z;          /* test function value  */
double *g, **h;    /* gradient and hessian matrix */
FILE *GKLS_info, *min_info, *fp, *fderiv; /* file pointers */
char filename[24]; /* name of files */
int probs_needed_to_get_100_feasible[10];
probs_needed_to_get_100_feasible[0] = 0;
probs_needed_to_get_100_feasible[1] = 0;
probs_needed_to_get_100_feasible[2] = 142;
probs_needed_to_get_100_feasible[3] = 214;
probs_needed_to_get_100_feasible[4] = 350;
probs_needed_to_get_100_feasible[5] = 716;
probs_needed_to_get_100_feasible[6] = 1313;
probs_needed_to_get_100_feasible[7] = 2780;

printf("\nGKLS-Generator of Classes of ND, D, and D2 Test Functions");
printf("\nfor Global Optimization,");
printf("\n(C) 2002-2003, M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev\n");


/*---------------------------------------------------------------------*/
/*  First, generate a class (with default parameters) of the D-type    */
/*  test functions and create the files with the function information  */
/*  and with the functions points (in the two-dimensional case).       */
/*---------------------------------------------------------------------*/

/* Set the input parameters */
/*if ((error_code=GKLS_set_default()) != GKLS_OK) {*/
/*  print_error_msg(error_code);*/
/*  return error_code;*/
/*}*/
  /* Another way to set the input parameters is:               */
for (i2=2; i2<=7; i2++){
  GKLS_dim = i2;                                             
  GKLS_num_minima = 10;                                     
  if ((error_code = GKLS_domain_alloc()) != GKLS_OK)        
    return error_code;                                     

  for (i=0; i<GKLS_dim; i++) {
    GKLS_domain_left[i]  = 0.0;
    GKLS_domain_right[i] = 1.0; 
  }
   
   GKLS_global_dist = sqrt(GKLS_dim)/2;
  /* GKLS_global_dist = 2.0/3.0;                               */
   GKLS_global_radius = 0.5*GKLS_global_dist;                
   GKLS_global_value = GKLS_GLOBAL_MIN_VALUE;                
   if ((error_code = GKLS_parameters_check()) != GKLS_OK)    
      return error_code;                                     

/* Allocate memory for the vector xx of feasible point */
if ( (xx=(double *)malloc((size_t)GKLS_dim*sizeof(double))) == NULL)
    return (-1);

/* Generate the class of 100 D-type functions */
for (func_num=1; func_num <= probs_needed_to_get_100_feasible[i2]; func_num++)
{
  if((error_code=GKLS_arg_generate (func_num)) != GKLS_OK) {
	print_error_msg(error_code);
	return error_code;
  }

  /* Open files */

  /* File of the test function information */
  /*sprintf(filename,"test_%d_%04d_%d.txt",GKLS_dim,func_num,GKLS_num_minima);*/
  /*if ((GKLS_info=fopen(filename,"wt")) == NULL) return (-1);*/

	/* File of local minimizers */
    /*sprintf(filename,"lmin%03d",func_num);*/
    sprintf(filename,"lmin_%d_%04d_%d",GKLS_dim,func_num,GKLS_num_minima);
 	if ((min_info=fopen(filename,"wt")) == NULL) return (-1);

  //if (GKLS_dim == 2) {

  //[> File of points <]
  //sprintf(filename,"test%03d",func_num);
  // if ((fp=fopen(filename,"wt")) == NULL) return (-1);

  //}

  printf("\nGenerating the function number %d\n", func_num);

  /*fprintf(GKLS_info,"D-type function number %d", func_num);*/
  /*fprintf(GKLS_info,"\nof the class with the following parameters:");*/
  /*fprintf(GKLS_info,"\n    problem dimension      = %u;",GKLS_dim);*/
  /*fprintf(GKLS_info,"\n    number of local minima = %u;",GKLS_num_minima);*/
  /*fprintf(GKLS_info,"\n    global minimum value   = %f;",GKLS_global_value);*/
  /*fprintf(GKLS_info,"\n    radius of the g.m.attraction region = %f;", GKLS_global_radius);*/
  /*fprintf(GKLS_info,"\n    distance from the paraboloid vertex to the global minimizer = %f.", GKLS_global_dist);*/

  /*[> Information about local minimizers <]*/
  /*fprintf(GKLS_info,"\n\nLocal minimizers:\n");*/
  /*for (i=0; i<GKLS_num_minima; i++) {*/
  /*  fprintf(GKLS_info,"  f[%u] = f(",i+1);*/
  /*  for (j=0; j<GKLS_dim; j++)*/
	/* fprintf(GKLS_info,"%7.3f",GKLS_minima.local_min[i][j]);*/
  /*  fprintf(GKLS_info,") = %7.3f;  ",GKLS_minima.f[i]);*/
	/*fprintf(GKLS_info,"rho[%u] = %7.3f.\n", i+1, GKLS_minima.rho[i]);*/
  /*}*/


  /*[> Information about global minimizers <]*/
  /*if (GKLS_glob.gm_index == 0)*/
	/*fprintf(GKLS_info,"\nAn error during the global minimum searching was occurred!");*/
  /*else {*/
  /*  if (GKLS_glob.num_global_minima == 1) {*/
	/*  fprintf(GKLS_info,"\n\nThere is one global minimizer.");*/
  /*    fprintf(GKLS_info,"\nThe number of the global minimizer is: ");*/
	/*}*/
	/*else {*/
	/*  fprintf(GKLS_info,"\n\nThere are %u global minimizers.",GKLS_glob.num_global_minima);*/
  /*    fprintf(GKLS_info,"\nThe numbers of the global minimizers are: ");*/
	/*}*/
  /*  for (i=0; i<GKLS_glob.num_global_minima; i++) {*/
	/*  fprintf(GKLS_info,"%u ",GKLS_glob.gm_index[i]+1);*/
	/*  [> Vector of coordinates of this minimizer is:               <]*/
	/*  [> double *x = GKLS_minima.local_min[GKLS_glob.gm_index[i]]  <]*/
	/*}*/
  /*}*/

  for (i=0; i<GKLS_num_minima; i++) {
    for (j=0; j<GKLS_dim; j++)
      fprintf(min_info,"%17.16f ",GKLS_minima.local_min[i][j]);
    fprintf(min_info,"%17.16f\n",GKLS_minima.f[i]);
  }
  //if (GKLS_dim == 2) {
  //  [> Get the files of the local minimizers points <]

  //[> Function evaluating in the grid 100x100 <]
  //for (dx1=GKLS_domain_left[0]; dx1<=GKLS_domain_right[0]+GKLS_PRECISION;
  //     dx1+=(GKLS_domain_right[0] - GKLS_domain_left[0])/100.0)
  //for (dx2=GKLS_domain_left[1]; dx2<=GKLS_domain_right[1]+GKLS_PRECISION;
  //       dx2+=(GKLS_domain_right[1] - GKLS_domain_left[1])/100.0)
  //  {
  //    xx[0]=dx1; xx[1]=dx2;
  //    z=GKLS_D_func(xx);
  //  [> z=GKLS_ND_func(xx); -- for ND-type test function <]
  //    [> z=GKLS_D2_func(xx); -- for D2-type test function <]
  //  if (z>=GKLS_MAX_VALUE-1000.0) [> An error: do something <];
  //  else
  //    fprintf(fp,"%f %f %f\n", dx1, dx2, z);
  //  }
  //} [> Creating files for two-dimensional functions <]

  //[> Close files <]
  //if (GKLS_dim == 2) {
  // fclose(fp); 
  //}
  fclose(min_info);
  /*fclose(GKLS_info);*/
  /* Deallocate memory */
  GKLS_free();
} /* for func_num*/
}

/*[> Deallocate memory of the vector xx of feasible point <]*/
/*  free(xx);*/

/*[>-----------------------------------------------------------------------<]*/
/*[>  Second, create the file with information about gradients of specific <]*/
/*[>  D-type and D2-type test functions and about Hessian matrix of a      <]*/
/*[>  specific D2-type test function at a given feasible point.            <]*/
/*[>-----------------------------------------------------------------------<]*/

/*[> Set the input parameters of the class <]*/
/*if ((error_code=GKLS_set_default()) != GKLS_OK) {*/
/*   print_error_msg(error_code);*/
/*   return error_code;*/
/*}*/

/*[> Allocate memory for the vector xx of feasible point <]*/
/*if ( (xx=(double *)malloc((size_t)GKLS_dim*sizeof(double))) == NULL)*/
/*    return (-1);*/

/* [> Generate a specific function from the class with default parameters <]*/
/* func_num = 9; [> test function number <]*/
/* if((error_code=GKLS_arg_generate (func_num)) != GKLS_OK) {*/
/*   print_error_msg(error_code);*/
/*   return error_code;*/
/* }*/

/* [> Open the file of gradients and Hessian matrix at a given point <]*/
/* sprintf(filename,"deriv%03d.txt",func_num);*/
/* if ((fderiv=fopen(filename,"wt")) == NULL) return (-1);*/

/* [> Allocate memory for the gradient vector and the Hessian matrix <]*/
/* if ( (g=(double *)malloc((size_t)GKLS_dim*sizeof(double))) == NULL)*/
/*    return (-1); [> gradient vector <]*/
/* if ( (h=(double **)malloc((size_t)GKLS_dim*sizeof(double *))) == NULL)*/
/*    return (-1); [> Hessian matrix <]*/
/* for (i=0; i<GKLS_dim; i++)*/
/*    if ( (h[i]=(double *)malloc((size_t)GKLS_dim*sizeof(double))) == NULL)*/
/*    return (-1);*/

/* [> Fix a trial point as the central point <]*/
/* for (i=0; i<GKLS_dim; i++) {*/
/*   xx[i] = 0.5*(GKLS_domain_right[i] + GKLS_domain_left[i]);*/
/* }*/

/* fprintf(fderiv,"Gradients and Hessian matrix at the point with coordinates:\n");*/
/* fprintf(fderiv,"   x = ( ");*/
/*  for (i=0; i<GKLS_dim; i++) {*/
/*     fprintf(fderiv,"%f ",xx[i]);*/
/* }*/
/* fprintf(fderiv,")\n\n");*/

/* [> Evaluate the gradient of the D-type function with the given number <]*/
/* if ((error_code=GKLS_D_gradient(xx,g)) != GKLS_OK) {*/
/*  print_error_msg(error_code);*/
/*  return error_code;*/
/* }*/

/* fprintf(fderiv,"\nVector of gradient of the D-type function number %d:\n",func_num);*/
/* fprintf(fderiv,"  grad D(x) = ( ");*/
/* for (i=0; i<GKLS_dim; i++) {*/
/*   fprintf(fderiv,"%f ",g[i]);*/
/* }*/
/* fprintf(fderiv,")\n");*/

/* [> Evaluate the gradient of the D2-type function with the given number <]*/
/* if ((error_code=GKLS_D2_gradient(xx,g)) != GKLS_OK) {*/
/*  print_error_msg(error_code);*/
/*  return error_code;*/
/* }*/

/* fprintf(fderiv,"\nVector of gradient of the D2-type function number %d:\n",func_num);*/
/* fprintf(fderiv,"  grad D2(x) = ( ");*/
/* for (i=0; i<GKLS_dim; i++) {*/
/*   fprintf(fderiv,"%f ",g[i]);*/
/* }*/
/* fprintf(fderiv,")\n");*/

/* [> Evaluate the Hessian matrix of the D2-type function with the given number <]*/
/* if ((error_code=GKLS_D2_hessian(xx,h)) != GKLS_OK) {*/
/*  print_error_msg(error_code);*/
/*  return error_code;*/
/* }*/

/* fprintf(fderiv,"\nHessian matrix of the D2-type function number %d:\n",func_num);*/
/* for (i=0; i<GKLS_dim; i++) {*/
/*     for (j=0; j<GKLS_dim; j++)*/
/*     fprintf(fderiv," %f ",h[i][j]);*/
/*   fprintf(fderiv,"\n");*/
/* }*/

/* [> Close file <]*/
/* fclose(fderiv);*/

/* [> Deallocate memory <]*/
/* free(g);*/
/* for (i=0; i<GKLS_dim; i++)*/
/*   free(h[i]);*/
/* free(h);*/

/* [> Deallocate memory of the vector xx of feasible point <]*/
/* free(xx);*/


 /* Deallocate the boundary vectors */
 GKLS_domain_free();

 return 0;

} /* example.c */
