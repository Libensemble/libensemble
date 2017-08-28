/*******************************************************************************/
/*        GKLS-Generator of Classes of ND  (non-differentiable),               */
/*                                 D  (continuously differentiable), and       */
/*                                 D2 (twice continuously differentiable)      */
/*        Test Functions for Global Optimization                               */
/*                                                                             */
/*   Authors:                                                                  */
/*                                                                             */
/*   M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                          */
/*                                                                             */
/*   (C) 2002-2005                                                             */
/*                                                                             */
/*   Each test class is defined by the following parameters:                   */
/*  (1) problem dimension                                                      */
/*  (2) number of local minima including paraboloid min and global min         */
/*  (3) global minimum value                                                   */
/*  (3) distance from the paraboloid vertex to the global minimizer            */
/*  (4) radius of the attraction region of the global minimizer                */
/*******************************************************************************/

/*******************************************************************************/
/*  The following groups of subroutines are present in the software:           */
/*   -- parameters setting/checking subroutines:                               */
/*       int  GKLS_set_default (void);                                         */
/*       int  GKLS_parameters_check (void);                                    */
/*   -- test classes generating subroutines:                                   */
/*       int  GKLS_arg_generate (unsigned int);                                */
/*       int  GKLS_coincidence_check (void);                                   */
/*       int  GKLS_set_basins (void);                                          */
/*   -- test functions calling subroutines:                                    */
/*       double GKLS_ND_func  (double *);                                      */
/*	     double GKLS_D_func   (double *);                                      */
/*		 double GKLS_D2_func  (double *);                                      */
/*   -- subroutines of evaluation of the partial derivatives of test functions */
/*       double GKLS_D_deriv   (unsigned int, double *);                       */
/*       double GKLS_D2_deriv1 (unsigned int, double *);                       */
/*       double GKLS_D2_deriv2 (unsigned int, unsigned int, double *);         */
/*   -- subroutines of evaluation of the gradients and hessian matrix          */
/*      int GKLS_D_gradient  (double *, double *);                             */
/*      int GKLS_D2_gradient (double *, double *);                             */
/*      int GKLS_D2_hessian  (double *, double **);                            */
/*   -- memory allocation/deallocation subroutines:                            */
/*	     int  GKLS_domain_alloc (void); / void GKLS_domain_free  (void);       */
/*		 int  GKLS_alloc (void); / void GKLS_free (void);                      */
/*   -- auxiliary subroutines                                                  */
/* 	     int    GKLS_initialize_rnd (unsigned int, unsigned int, int);         */
/*       double GKLS_norm (double *, double *);                                */
/*******************************************************************************/

#include "gkls.h"
#include "rnd_gen.h"
#include <math.h>
#include <stdlib.h>
// #include <malloc.h>

/*---------------- Variables accessible by the user -------------------- */
double *GKLS_domain_left; /* left boundary vector of D  */
  /* D=[GKLS_domain_left; GKLS_domain_ight] */
double *GKLS_domain_right;/* right boundary vector of D */

unsigned int GKLS_dim;    /* dimension of the problem,        */
                          /* 2<=GKLS_dim<NUM_RND (see random) */
unsigned int GKLS_num_minima; /* number of local minima, >=2  */

double GKLS_global_dist;  /* distance from the paraboloid minimizer  */
                          /* to the global minimizer                 */
double GKLS_global_radius;/* radius of the global minimizer          */
                          /* attraction region                       */
double GKLS_global_value; /* global minimum value,                   */
                          /* GKLS_global_value < GKLS_PARABOLOID_MIN */
T_GKLS_Minima GKLS_minima;
                          /* see the structures type description     */
T_GKLS_GlobalMinima GKLS_glob;

/*--------------------------- Global variables ----------------------*/
int isArgSet=0; /* isArgSet == 1 if all necessary parameters are set */

double delta; /* parameter using in D2-type function generation;     */
              /* it is chosen randomly from the                      */
              /* open interval (0,GKLS_DELTA_MAX_VALUE)              */
unsigned long rnd_counter; /* index of random array elements */

/*------------------ Auxiliary functions prototypes -----------------*/

double GKLS_norm(double *, double *);
int GKLS_alloc(void);
int GKLS_coincidence_check(void);
int GKLS_set_basins(void);
int GKLS_initialize_rnd(unsigned int, unsigned int, int);

/*****************************************************************************/
/*    Distance between two vectors in the Euclidean space R^(GKLS_dim)       */
/* INPUT:                                                                    */
/*    x1, x2 -- arrays of the coordinates of the two vectors x1 and x2       */
/*              of the dimension GKLS_dim                                    */
/* RETURN VALUE: Euclidean norm ||x1-x2||                                    */
/*****************************************************************************/
double GKLS_norm(double *x1, double *x2)
{
 unsigned int i;
 double norm = 0.0;
 for (i=0; i<GKLS_dim; i++)
   norm += (x1[i] - x2[i])*(x1[i] - x2[i]);
 return sqrt(norm);
} /* GKLS_norm() */

/*****************************************************************************/
/*     Allocating memory for the boundary vectors of the admissible region   */
/*     and setting (by default) D=[-1,1]^(GKLS_dim)                          */
/*     This subroutine should be called before the work with the generator   */
/* The subroutine has no INPUT parameters                                    */
/* RETURN VALUE: an error code                                               */
/*****************************************************************************/
int GKLS_domain_alloc()
{
 unsigned int i;

 if ( (GKLS_dim <= 1) || (GKLS_dim >= NUM_RND) )
   return GKLS_DIM_ERROR; /* problem dimension error */
 if ((GKLS_domain_left = (double *)(malloc((size_t)GKLS_dim*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR; /* memory allocation error */
 if ((GKLS_domain_right =(double *)(malloc((size_t)GKLS_dim*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR; /* memory allocation error */
 /* Set the admissible region as [-1,1]^GKLS_dim */
 for (i=0; i<GKLS_dim; i++) {
   GKLS_domain_left[i]  = -1.0;
   GKLS_domain_right[i] =  1.0;
 }
 return GKLS_OK; /* no errors */
} /* GKLS_domain_alloc() */

/*****************************************************************************/
/* Deallocating memory allocated for the boundary vectors                    */
/* This subroutine should be called at the end of the work with the generator*/
/*****************************************************************************/
void GKLS_domain_free()
{
 free(GKLS_domain_left);
 free(GKLS_domain_right);
} /* GKLS_domain_free() */


/*****************************************************************************/
/*   Setting default values of the input parameters                          */
/*   If the boundary vectors have not been allocated,                        */
/*   the subroutine allocates them                                           */
/* The subroutine has no INPUT parameters                                    */
/* RETURN VALUE: an error code (result of the operation of memory allocation */
/*****************************************************************************/
int GKLS_set_default()
{
  unsigned int i;
  int error;
  double min_side, tmp;

  GKLS_dim = 2;

  GKLS_num_minima = 10;

  if ((GKLS_domain_left == NULL) || (GKLS_domain_right == NULL)) {
	  /* define the boundaries  */
	  if ((error=GKLS_domain_alloc()) != GKLS_OK) return error;
  }
  /* Find min_side = min |b(i)-a(i)|, D=[a,b], and                       */
  /* set the distance from the paraboloid vertex to the global minimizer */
  min_side = GKLS_domain_right[0]-GKLS_domain_left[0];
  for (i=1; i<GKLS_dim; i++)
    if ((tmp=GKLS_domain_right[i]-GKLS_domain_left[i]) < min_side)
		min_side = tmp;
  GKLS_global_dist = min_side/3.0;

  GKLS_global_radius = 0.5*GKLS_global_dist;

  GKLS_global_value = GKLS_GLOBAL_MIN_VALUE;

  return GKLS_OK;

} /* GKLS_set_default() */


/*****************************************************************************/
/*     Allocating memory for dynamic arrays                                  */
/*     (for lists of the structures GKLS_minima and GKLS_glob)               */
/*     It is called by the generator subroutine GKLS_arg_generator()         */
/* The subroutine has no INPUT parameters                                    */
/* RETURN VALUE: an error code                                               */
/*****************************************************************************/
int GKLS_alloc()
{
 unsigned int i;

 if ( (GKLS_dim <= 1) || (GKLS_dim >= NUM_RND) )
   return GKLS_DIM_ERROR; /* problem dimension error */
 if (GKLS_num_minima <= 1)
   return GKLS_NUM_MINIMA_ERROR; /* erroneous number of local minima */
 if ((GKLS_minima.local_min =(double **)(malloc((size_t)GKLS_num_minima*sizeof(double *)))) == NULL)
   return GKLS_MEMORY_ERROR; /* memory allocation error */
 for (i=0; i<GKLS_num_minima; i++)
   if ((GKLS_minima.local_min[i] =(double *)(malloc((size_t)GKLS_dim*sizeof(double)))) == NULL)
     return GKLS_MEMORY_ERROR; /* memory allocation error */
 if ((GKLS_minima.w_rho =(double *)(malloc((size_t)GKLS_num_minima*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR;   /* memory allocation error */
 if ((GKLS_minima.peak =(double *)(malloc((size_t)GKLS_num_minima*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR;   /* memory allocation error */
 if ((GKLS_minima.rho =(double *)(malloc((size_t)GKLS_num_minima*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR;   /* memory allocation error */
 if ((GKLS_minima.f =(double *)(malloc((size_t)GKLS_num_minima*sizeof(double)))) == NULL)
   return GKLS_MEMORY_ERROR;   /* memory allocation error */
 if ((GKLS_glob.gm_index =
	 (unsigned int *)(malloc((size_t)GKLS_num_minima*sizeof(unsigned int)))) == NULL)
   return GKLS_MEMORY_ERROR;   /* memory allocation error */
 else
   GKLS_glob.num_global_minima = 0;

 return GKLS_OK; /* no errors */
} /* GKLS_alloc() */

/*******************************************************************************/
/* Deallocating dynamic memory allocated for structures GKLS_minima / GKLS_glob*/
/* It should be called at the end of the work with a generated test function   */
/*******************************************************************************/
void GKLS_free()
{
 unsigned int i;

 for (i=0; i<GKLS_num_minima; i++) {
	 free(GKLS_minima.local_min[i]);
 }
 free(GKLS_minima.local_min);
 free(GKLS_minima.w_rho);
 free(GKLS_minima.peak);
 free(GKLS_minima.rho);
 free(GKLS_minima.f);
 free(GKLS_glob.gm_index);

 isArgSet = 0; /* Parameters do not exist more */
} /* GKLS_free() */


/****************************************************************************/
/*           Checking the input parameters                                  */
/* The subroutine has no INPUT parameters                                   */
/* RETURN VALUE: an error code if there is an erroneous parameter           */
/****************************************************************************/
int GKLS_parameters_check()
{
 unsigned int i;
 double min_side, tmp;

 if  ((GKLS_dim <= 1) || (GKLS_dim >= NUM_RND))
   return GKLS_DIM_ERROR;   /* problem dimension errors */
 if (GKLS_num_minima <= 1)  /* number of local minima error */
	 return GKLS_NUM_MINIMA_ERROR;
 if ((GKLS_domain_left == NULL) || (GKLS_domain_right == NULL))
	 return GKLS_BOUNDARY_ERROR; /* the boundaries are not defined */
 for (i=0; i<GKLS_dim; i++)
  if (GKLS_domain_left[i] >= GKLS_domain_right[i] - GKLS_PRECISION)
     return GKLS_BOUNDARY_ERROR; /* the boundaries are erroneous */
 if (GKLS_global_value >= GKLS_PARABOLOID_MIN - GKLS_PRECISION)
	 return GKLS_GLOBAL_MIN_VALUE_ERROR; /* the global minimum value must   */
                                         /* be less than the paraboloid min */
 /* Find min_side = min |b(i)-a(i)|, D=[a,b], and                   */
 /* check the distance from paraboloid vertex to global minimizer   */
 min_side = GKLS_domain_right[0]-GKLS_domain_left[0];
 for (i=1; i<GKLS_dim; i++)
    if ((tmp=GKLS_domain_right[i]-GKLS_domain_left[i]) < min_side)
		min_side = tmp;
 /*if ( (GKLS_global_dist >= 0.5*min_side - GKLS_PRECISION) ||*/
 /*     (GKLS_global_dist <= GKLS_PRECISION) )*/
 /*   return GKLS_GLOBAL_DIST_ERROR; [> global distance error <]*/
 if ( (GKLS_global_radius >= 0.5*GKLS_global_dist + GKLS_PRECISION) ||
	  (GKLS_global_radius <= GKLS_PRECISION) )
	return GKLS_GLOBAL_RADIUS_ERROR; /* global minimizer attr. radius error */

 return GKLS_OK; /* no errors */
} /* GKLS_parameters_check() */


/*****************************************************************************/
/*       The subroutine checks possible coincidence of local minimizers      */
/* The subroutine has no INPUT parameters                                    */
/* RETURN VALUE: an error code (GKLS_OK if there are no errors):             */
/*   GKLS_PARABOLA_MIN_COINCIDENCE_ERROR - if some local minimizer coincides */
/*                                         with the paraboloid minimizer     */
/*   GKLS_LOCAL_MIN_COINCIDENCE_ERRO     - if there is a pair of identical   */
/*                                         local minimizers                  */
/*****************************************************************************/
int GKLS_coincidence_check()
{
 unsigned int i, j;

 /* Check wether some local minimizer coincides with the paraboloid minimizer */
 for (i=2; i<GKLS_num_minima; i++)
  {
   if ( GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[0]) < GKLS_PRECISION )
    return GKLS_PARABOLA_MIN_COINCIDENCE_ERROR;
  }

 /* Check wether there is a pair of identical local minimizers */
 for (i=1; i<GKLS_num_minima-1; i++)
  for (j=i+1; j<GKLS_num_minima; j++)
  {
   if ( GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[j]) < GKLS_PRECISION )
    return GKLS_LOCAL_MIN_COINCIDENCE_ERROR;
  }

 return GKLS_OK;

} /* GKLS_coincidence_check() */

/*****************************************************************************/
/*    The subroutine determines attraction regions of local minimizers       */
/* It has no INPUT parameters                                                */
/* RETURN VALUE: an error code                                               */
/*****************************************************************************/
int GKLS_set_basins()
{
 unsigned int i, j;
 double temp_min;         /*  temporary  */
 double temp_d1, temp_d2; /*  variables  */
 double dist;  /* for finding the distance between two minimizers */

 /****************************************************************************/
 /* First, set the radii rho(i) of the attraction regions: these values are  */
 /* defined in such a way that attraction regions are as large as possible   */
 /* and do not overlap; it is not required that the attraction region of each*/
 /* local minimizer be entirely contained in D. The values found in such     */
 /* a manner are corrected then by the weight coefficients w(i)              */
 /****************************************************************************/

 /* Calculate dist(i) - the minimal distance from the minimizer i to         */
 /*                     the other minimizers.                                */
 /* Set the initial value of rho(i) as rho(i) = dist(i)/2: so the attraction */
 /* regions do not overlap                                                   */
 for (i=0; i<GKLS_num_minima; i++)
  {
   temp_min = GKLS_MAX_VALUE;
   for (j=0; j<GKLS_num_minima; j++)
    if (i != j)
     {
      if ((temp_d1=GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[j])) < temp_min)
        temp_min = temp_d1;
     }

   dist = temp_min / 2.0;

   GKLS_minima.rho[i] = dist;
 }

 /* Since the radius of the attraction region of the global minimizer            */
 /* is fixed by the user, the generator adjusts the radii of the attraction      */
 /* regions, eventually overlapping with the attraction region of the global     */
 /* minimizer. To do this, it checks whether the attraction region radius        */
 /* of each local minimizer exceeds the distance between this minimizer          */
 /* and the attraction region of the global minimizer.                           */
 /* If such a situation is verified the generator decreases the attraction       */
 /* region radius of the local minimizer setting it equal to he distance between */
 /* the local minimizer and the attraction region of the global minimizer.       */
 /* Note that the radius of the attraction region of the global minimizer can    */
 /* not be greater than one half of the distance (defined by the user) between   */
 /* the global minimizer and the paraboloid vertex. So, the initially defined    */
 /* attraction regions of the global minimizer and the paraboloid vertex do not  */
 /* overlap even when the global minimizer is the closest minimizer to the       */
 /* paraboloid vertex.                                                           */
  GKLS_minima.rho[1] = GKLS_global_radius;
  for (i=2; i<GKLS_num_minima; i++)
  {
    if ((dist=(GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[1])
		      -GKLS_global_radius-GKLS_PRECISION)) < GKLS_minima.rho[i])
	   GKLS_minima.rho[i] = dist;
  }
 /* Try to expand the attraction regions of local minimizers until they      */
 /* do not overlap                                                           */
 for (i=0; i<GKLS_num_minima; i++)
 {
   if (i != 1) /* The radius of the attr. region of the global min is fixed  */
   { /* rho(i) := max {rho(i),min[||M(i)-M(j)|| - rho(j): i !=j]},      */
	 temp_min = GKLS_MAX_VALUE;
     for (j=0; j<GKLS_num_minima; j++)
     if (i != j)
	 {
      if ((temp_d1=GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[j]) - GKLS_minima.rho[j]) < temp_min)
        temp_min = temp_d1;
     }
     /* Increase the radius rho(i) if it is possible */
	 if (temp_min > GKLS_minima.rho[i] + GKLS_PRECISION)
	   GKLS_minima.rho[i] = temp_min;
   }
 }

 /* Correct the radii by weight coefficients w(i)                    */
 /* The weight coefficients can be chosen randomly;                  */
 /* here they are defined by default as:                             */
 /*    w(i) = 0.99, i != 1 , and w(1) = 1.0 (global min index = 1)   */
 for (i=0; i<GKLS_num_minima; i++)
     GKLS_minima.rho[i] = GKLS_minima.w_rho[i] * GKLS_minima.rho[i];

 /*********************************************************************/
 /* Set the local minima values f(i) of test functions as follows:    */
 /*   f(i) = cond_min(i) - peak(i), i != 1 (global min index = 1)     */
 /*   f(0) = GKLS_PARABOLOID_MIN, f(1) = GKLS_GLOBAL_MIN_VALUE,       */
 /* where cond_min(i) is the paraboloid minimum value at the boundary */
 /* B={||x-M(i)||=rho(i)} of the attraction region of the local       */
 /* minimizer M(i), i.e.                                              */
 /*  cond_min(i) =                                                    */
 /*  = paraboloid g() value at (M(i)+rho(i)*(T-M(i))/norm(T-M)) =     */
 /*  = (rho(i) - norm(T-M(i)))^2 + t,                                 */
 /*  g(x) = ||x-T||^2 + t, x in D of R^GKLS_dim                       */
 /*                                                                   */
 /*  The values of peak(i) are chosen randomly from an interval       */
 /* (0, 2*rho(i), so that the values f(i) depend on radii rho(i) of   */
 /* the attraction regions, 2<=i<GKLS_dim.                            */
 /*  The condition f(x*)=f(1) <= f(i) must be satisfied               */
 /*********************************************************************/
 /* Fix to 0 the peak(i) values of the paraboloid and the global min  */
 GKLS_minima.peak[0] = 0.0; GKLS_minima.peak[1] = 0.0;
 for (i=2; i<GKLS_num_minima; i++) {
  /* Set values peak(i), i>= 2 */
  /* Note that peak(i) is such that the function value f(i) is smaller*/
  /* than min(GKLS_GLOBAL_MIN_VALUE, 2*rho(i))                        */
  temp_d1=GKLS_norm(GKLS_minima.local_min[0], GKLS_minima.local_min[i]);
  temp_min=(GKLS_minima.rho[i] - temp_d1)*(GKLS_minima.rho[i] - temp_d1)+
           GKLS_minima.f[0]; /*the conditional minimum at the boundary*/

  temp_d1 = (1.0 + rnd_num[rnd_counter])*GKLS_minima.rho[i];
  temp_d2 = rnd_num[rnd_counter]*(temp_min - GKLS_global_value);
  /* temp_d1 := min(temp_d1, temp_d2) */
  if (temp_d2 < temp_d1) temp_d1 = temp_d2;
  GKLS_minima.peak[i]= temp_d1;

  rnd_counter++;
  if (rnd_counter == NUM_RND)
    { ranf_array(rnd_num, NUM_RND); rnd_counter = 0L; }

  GKLS_minima.f[i] = temp_min - GKLS_minima.peak[i];
 }

/*********************************************************************/
/* Find all possible global minimizers and                           */
/* create a list of their indices among all the minimizers           */
/* Note that the paraboloid minimum can not be the global one because*/
/* the global optimum value is set to be less than the paraboloid    */
/* minimum value                                                     */
/*********************************************************************/
 GKLS_glob.num_global_minima = 0;
 for (i=0; i<GKLS_num_minima; i++)
	if ((GKLS_minima.f[i] >= GKLS_global_value - GKLS_PRECISION) &&
		(GKLS_minima.f[i] <= GKLS_global_value + GKLS_PRECISION))
	{
	   GKLS_glob.gm_index[GKLS_glob.num_global_minima] = i;
	   GKLS_glob.num_global_minima ++;
	/* The first GKLS_glob.num_global_minima elements of the list    */
	/* contain the indices of the global minimizers                  */
	}
    else
       GKLS_glob.gm_index[GKLS_num_minima-1-i+GKLS_glob.num_global_minima]
	    = i;
	/* The remaining elements of the list                            */
	/* contain the indices of local (non global) minimizers          */

 if (GKLS_glob.num_global_minima == 0) /*   erroneous case:       */
	return GKLS_FLOATING_POINT_ERROR;  /* some programmer's error */

 return GKLS_OK;
} /* GKLS_set_basins() */

/*****************************************************************************/
/* The subroutine initializes random sequence by generating a seed that      */
/* depends on a specific test function number, on the number of local minima,*/
/* and on the problem dimension                                              */
/* INPUT PARAMETERS:                                                         */
/*  dim  -- dimension of the problem (dim >= 2);                             */
/*  nmin -- number of local minima (nmin >= 2);                              */
/*  nf   -- test function number (from 1 to 100)                             */
/* RETURN VALUE: normally GKLS_OK                                            */
/*****************************************************************************/
int GKLS_initialize_rnd(unsigned int dim, unsigned int nmin, int nf)
{
 long seed;
 /* seed number between 0 and 2^30-3 = 1,073,741,821*/

 seed = (nf-1) + (nmin-1)*100 + dim*1000000L;
 /* If big values of nmin and dim are required, */
 /* one must check wether seed <= 1073741821    */

 ranf_start(seed);

 return GKLS_OK;
} /* GKLS_initialize_rnd() */


/*****************************************************************************/
/* The main subroutine of the package that generates randomly the local and  */
/* the global minimizers and function values at minimizers;                  */
/* it determines the radii of attraction regions of the minimizers           */
/* INPUT PARAMETER:                                                          */
/*   nf -- determines the number of test function, 1 <= nf <= 100            */
/* RETURN VALUE:                                                             */
/*   an error code                                                           */
/* The boundaries vectors should be created and                              */
/* the parameters of the class should be defined first                       */
/*****************************************************************************/
int GKLS_arg_generate (unsigned int nf)
{
 unsigned int i, j;
 int error;
 double sin_phi; /* for generating of the global minimizer coordinates */
                 /* by using the generalized spherical coordinates     */
 double gap = GKLS_global_radius; /* gap > 0 */
       /* the minimal distance of any local minimizer to the attraction*/
       /* region of the global minimizer M(1); the value               */
       /* GKLS_global_radius is given by default and can be changed,   */
       /* but it should not be too small.                              */

 /* Check function number */
 if ((nf < 1) || (nf > 4000)) return GKLS_FUNC_NUMBER_ERROR;

 /* Check parameters */
 if ( (error = GKLS_parameters_check()) != GKLS_OK) return error;

 /* Allocate memory */
 if ( (error = GKLS_alloc()) != GKLS_OK) return error;

 /* Set random seed */
 if ( (error = GKLS_initialize_rnd(GKLS_dim,GKLS_num_minima,nf)) != GKLS_OK)
    return error;

 ranf_array(rnd_num, NUM_RND); /* get random sequence */
 rnd_counter = 0L;   /* index of the random element from the sequence */
                     /* to be used as the next random number          */

 /* Set the paraboloid minimizer coordinates and */
 /* the paraboloid minimum value                 */
 for (i=0; i<GKLS_dim; i++) {
   GKLS_minima.local_min[0][i] = GKLS_domain_left[i] +
		   rnd_num[rnd_counter]*(GKLS_domain_right[i] - GKLS_domain_left[i]);
   rnd_counter++;
   if (rnd_counter == NUM_RND) {
     ranf_array(rnd_num, NUM_RND); rnd_counter = 0L;
   }
 } /* for coordinates */
 GKLS_minima.f[0] = GKLS_PARABOLOID_MIN; /* fix the paraboloid min value */

 /* Generate the global minimizer using generalized spherical coordinates*/
 /* with an arbitrary vector phi and the fixed radius GKLS_global_radius */

 /* First, generate an angle 0 <= phi(0) <= PI, and the coordinate x(0)*/
   ranf_array(rnd_num, NUM_RND);
   rnd_counter = 0L;
   GKLS_minima.local_min[1][0] = GKLS_minima.local_min[0][0] +
	       GKLS_global_dist*cos(PI*rnd_num[rnd_counter]);
   if ( (GKLS_minima.local_min[1][0] > GKLS_domain_right[0] - GKLS_PRECISION) ||
	    (GKLS_minima.local_min[1][0] < GKLS_domain_left[0] + GKLS_PRECISION) )
      GKLS_minima.local_min[1][0] = GKLS_minima.local_min[0][0] -
	       GKLS_global_dist*cos(PI*rnd_num[rnd_counter]);
   sin_phi = sin(PI*rnd_num[rnd_counter]);
   rnd_counter++;

 /* Generate the remaining angles 0<=phi(i)<=2*PI, and         */
 /* the coordinates x(i), i=1,...,GKLS_dim-2 (not last!)       */
   for(j=1; j<GKLS_dim-1; j++) {
    GKLS_minima.local_min[1][j] = GKLS_minima.local_min[0][j] +
		   GKLS_global_dist*cos(2.0*PI*rnd_num[rnd_counter])*sin_phi;
    if ( (GKLS_minima.local_min[1][j] > GKLS_domain_right[j] - GKLS_PRECISION) ||
		 (GKLS_minima.local_min[1][j] < GKLS_domain_left[j] + GKLS_PRECISION) )
       GKLS_minima.local_min[1][j] = GKLS_minima.local_min[0][j] -
	       GKLS_global_dist*cos(2.0*PI*rnd_num[rnd_counter])*sin_phi;
    sin_phi *= sin(2.0*PI*rnd_num[rnd_counter]);
    rnd_counter++;
   }

 /* Generate the last coordinate x(GKLS_dim-1) */
   GKLS_minima.local_min[1][GKLS_dim-1] = GKLS_minima.local_min[0][GKLS_dim-1] +
	       GKLS_global_dist*sin_phi;
   if ( (GKLS_minima.local_min[1][GKLS_dim-1] > GKLS_domain_right[GKLS_dim-1] - GKLS_PRECISION) ||
	    (GKLS_minima.local_min[1][GKLS_dim-1] < GKLS_domain_left[GKLS_dim-1] + GKLS_PRECISION) )
      GKLS_minima.local_min[1][GKLS_dim-1] =
	      GKLS_minima.local_min[0][GKLS_dim-1] - GKLS_global_dist*sin_phi;

 /* Set the global minimum value */
 GKLS_minima.f[1] = GKLS_global_value;

 /* Set the weight coefficients w_rho(i) */
 for (i=0; i<GKLS_num_minima; i++)
    GKLS_minima.w_rho[i] = 0.99;
 GKLS_minima.w_rho[1]=1.0;


 /* Set the parameter delta for D2-type functions       */
 /* It is chosen randomly from (0,GKLS_DELTA_MAX_VALUE) */
 delta = GKLS_DELTA_MAX_VALUE*rnd_num[rnd_counter];
 rnd_counter++;
 if (rnd_counter == NUM_RND) {
	 ranf_array(rnd_num, NUM_RND); rnd_counter = 0L;
 }

 /* Choose randomly coordinates of local minimizers       */
 /* This procedure is repeated while the local minimizers */
 /* coincide (external do...while);                       */
 /* The internal cycle do..while serves to choose local   */
 /* minimizers in certain distance from the attraction    */
 /* region of the global minimizer M(i)                   */
 do
 {
   i=2;
   while (i<GKLS_num_minima) {
    do
    {
	  ranf_array(rnd_num, NUM_RND);
      rnd_counter = 0L;
      for(j=0; j<GKLS_dim; j++) {
	    GKLS_minima.local_min[i][j] = GKLS_domain_left[j] +
		  rnd_num[rnd_counter]*(GKLS_domain_right[j] - GKLS_domain_left[j]);
	    rnd_counter++;
	    if (rnd_counter == NUM_RND) {
		   ranf_array(rnd_num, NUM_RND); rnd_counter = 0L;
		}
	  }
     /* Check wether this local minimizer belongs to a zone of */
	 /* the global minimizer M(i)                              */
    } while ( (GKLS_global_radius + gap) -
		      GKLS_norm(GKLS_minima.local_min[i], GKLS_minima.local_min[1])
		       > GKLS_PRECISION );
   i++;
   }
   error = GKLS_coincidence_check();
 } while ( (error == GKLS_PARABOLA_MIN_COINCIDENCE_ERROR) ||
           (error == GKLS_LOCAL_MIN_COINCIDENCE_ERROR) );
 error = GKLS_set_basins();
 if (error == GKLS_OK) isArgSet = 1; /* All the parameters are set */
 /* and the user can evaluate a specific test function or          */
 /* its partial derivative by calling corresponding subroutines    */

 return error;

} /* GKLS_arg_generate () */


/************************************************************************/
/*  The subroutine evaluates the generated function                     */
/*  of the ND-type (non-differentiable)                                 */
/*                                                                      */
/* INPUT PARAMETER:                                                     */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space         */
/* RETURN VALUE:                                                        */
/*   a function value OR                                                */
/*   GKLS_MAX_VALUE if: (1) the vector x does not belong to D;          */
/*                      (2) the user tries to call the function without */
/*                          parameter defining                          */
/************************************************************************/
double GKLS_ND_func(double *x)
{
 unsigned int i, index;
 double norm, scal, a, rho; /* working variables */

 if (!isArgSet) return GKLS_MAX_VALUE;

 for (i=0; i<GKLS_dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) || (x[i] > GKLS_domain_right[i]+GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima, M(index) <> T */
 /* Attention: number of local minima must be >= 2 */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
  {
   norm = GKLS_norm(GKLS_minima.local_min[0],x);
   /* Return the value of the paraboloid function */
   return (norm * norm + GKLS_minima.f[0]);
  }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
    return GKLS_minima.f[index];

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<GKLS_dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 /* Return the value of the quadratic interpolation function */
 return ((1.0 - 2.0/rho * scal / norm + a/rho/rho)*norm*norm +
	     GKLS_minima.f[index]);

} /* GKLS_ND_func() */


/************************************************************************/
/*  The subroutine evaluates the generated function                     */
/*  of the D-type (continuously differentiable)                         */
/*                                                                      */
/* INPUT PARAMETER:                                                     */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space         */
/* RETURN VALUE:                                                        */
/*   a function value OR                                                */
/*   GKLS_MAX_VALUE if: (1) the vector x does not belong to D;          */
/*                      (2) the user tries to call the function without */
/*                          parameter defining                          */
/************************************************************************/
double GKLS_D_func(double *x)
{
 unsigned int i, index;
 double norm, scal, a, rho; /* working variables */

 if (!isArgSet) return GKLS_MAX_VALUE;

 for (i=0; i<GKLS_dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) || (x[i] > GKLS_domain_right[i]+GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima, M(index) <> T */
 /* Attention: number of local minima must be >= 2 */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
  {
   norm = GKLS_norm(GKLS_minima.local_min[0],x);
   /* Return the value of the paraboloid function */
   return (norm * norm + GKLS_minima.f[0]);
  }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
    return GKLS_minima.f[index];

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<GKLS_dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 /* Return the value of the cubic interpolation function */
 return (2.0/rho/rho * scal / norm - 2.0*a/rho/rho/rho)*norm*norm*norm +
        (1.0-4.0*scal/norm/rho + 3.0*a/rho/rho)*norm*norm + GKLS_minima.f[index];

} /* GKLS_D_func() */


/************************************************************************/
/*  The subroutine evaluates the generated function                     */
/*  of the D2-type (twice continuously differentiable)                  */
/*                                                                      */
/* INPUT PARAMETER:                                                     */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space         */
/* RETURN VALUE:                                                        */
/*   a function value OR                                                */
/*   GKLS_MAX_VALUE if: (1) the vector x does not belong to D;          */
/*                      (2) the user tries to call the function without */
/*                          parameter defining                          */
/************************************************************************/
double GKLS_D2_func(double *x)
{
 unsigned int dim, i, index;
 double norm, scal, a, rho; /* working variables */

 if (!isArgSet) return GKLS_MAX_VALUE;

 dim = GKLS_dim;
 for (i=0; i<dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) ||
	   (x[i] > GKLS_domain_right[i] + GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
  {
   norm = GKLS_norm(GKLS_minima.local_min[0],x);
   /* Return the value of the paraboloid function */
   return (norm * norm + GKLS_minima.f[0]);
  }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
    return GKLS_minima.f[index];

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 /* Return the value of the quintic interpolation function */
 return ( (-6.0*scal/norm/rho + 6.0*a/rho/rho + 1.0 - delta/2.0) *
                                 norm * norm / rho / rho  +
          (16.0*scal/norm/rho - 15.0*a/rho/rho - 3.0 + 1.5*delta) * norm/rho +
          (-12.0*scal/norm/rho + 10.0*a/rho/rho + 3.0 - 1.5*delta)) *
          norm * norm * norm / rho +
          0.5*delta*norm*norm + GKLS_minima.f[index];

} /* GKLS_D2_func() */


/*******************************************************************************/
/*  The subroutine evaluates the first order partial derivative of D-type      */
/*  function with respect to the variable indicated by the user                */
/* INPUT PARAMETERS:                                                           */
/*   var_j -- an index of the variable with respect to which the derivative    */
/*            is evaluated                                                     */
/*   x     -- a point of the (GKLS_dim)-dimensional euclidean space            */
/* RETURN VALUE:                                                               */
/*   a first order partial derivative value OR                                 */
/*   GKLS_MAX_VALUE if: (1) the index of variable is out of range [1,GKLS_dim];*/
/*                      (2) the vector x does not belong to D;                 */
/*                      (3) the user tries to call the function without        */
/*                          parameter defining                                 */
/*******************************************************************************/
double GKLS_D_deriv(unsigned int var_j, double *x)
{
 unsigned int i, index;
 double norm, scal, dif, a, rho, h; /* working variables */

 if ( (var_j == 0) || (var_j > GKLS_dim) ) return GKLS_MAX_VALUE;
 else  var_j = var_j - 1; /* to be used as an index of array */

 if (!isArgSet) return GKLS_MAX_VALUE;

 for (i=0; i<GKLS_dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) || (x[i] > GKLS_domain_right[i]+GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima, M(index) <> T */
 /* Attention: number of local minima must be >= 2 */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
  {
   /* Return the value of the first order partial derivative of the paraboloid function */
   return 2.0*(x[var_j]-GKLS_minima.local_min[0][var_j]);
  }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
    return 0.0;

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<GKLS_dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 dif = x[var_j] - GKLS_minima.local_min[index][var_j];
 h = (GKLS_minima.local_min[0][var_j]-GKLS_minima.local_min[index][var_j])*norm -
     scal*dif/norm;
 /* Return the value of dC(x)/dx[var_i] of the D-type function */
 return ( h * (2.0/rho/rho*norm - 4.0/rho) +
	      dif * (6.0/rho/rho*scal - 6.0/rho/rho/rho*a*norm -
		         8.0/rho/norm*scal + 6.0/rho/rho*a + 2.0) );

} /* GKLS_D_deriv() */

/*******************************************************************************/
/*  The subroutine evaluates the first order partial derivative of D2-type     */
/*  function with respect to the variable indicated by the user                */
/* INPUT PARAMETERS:                                                           */
/*   var_j -- an index of the variable with respect to which the derivative    */
/*            is evaluated                                                     */
/*   x     -- a point of the (GKLS_dim)-dimensional euclidean space            */
/* RETURN VALUE:                                                               */
/*   a first order partial derivative value OR                                 */
/*   GKLS_MAX_VALUE if: (1) the index of variable is out of range [1,GKLS_dim];*/
/*                      (2) the vector x does not belong to D;                 */
/*                      (3) the user tries to call the function without        */
/*                          parameter defining                                 */
/*******************************************************************************/
double GKLS_D2_deriv1(unsigned int var_j, double *x)
{
 unsigned int i, index;
 double norm, scal, dif, a, rho, h; /* working variables */

 if ( (var_j == 0) || (var_j > GKLS_dim) ) return GKLS_MAX_VALUE;
 else  var_j = var_j - 1; /* to be used as an index of array */

 if (!isArgSet) return GKLS_MAX_VALUE;

 for (i=0; i<GKLS_dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) || (x[i] > GKLS_domain_right[i]+GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima, M(index) <> T */
 /* Attention: number of local minima must be >= 2 */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
  {
   /* Return the value of the first order partial derivative of the paraboloid function */
   return 2.0*(x[var_j]-GKLS_minima.local_min[0][var_j]);
  }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
    return 0.0;

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<GKLS_dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 dif = x[var_j] - GKLS_minima.local_min[index][var_j];
 h = (GKLS_minima.local_min[0][var_j]-GKLS_minima.local_min[index][var_j])*norm -
     scal*dif/norm;
 /* Return the value of dQ(x)/dx[var_i] of the D2-type function */
 return ( h*norm/rho/rho * (-6.0*norm*norm/rho/rho + 16.0*norm/rho - 12.0) +
	dif*norm * ((-30.0/rho/norm*scal+30/rho/rho*a+5.0-2.5*delta)/rho/rho/rho*norm*norm +
	            (64.0/rho/norm*scal-60.0/rho/rho*a-12.0+6.0*delta)/rho/rho*norm +
	            (-36.0/rho/norm*scal+30.0/rho/rho*a+9.0-4.5*delta)/rho) +
	dif*delta );


} /* GKLS_D2_deriv1() */


/*******************************************************************************/
/*  The subroutine evaluates the second order partial derivative of D2-type    */
/*  function with respect to the variables indicated by the user               */
/* INPUT PARAMETERS:                                                           */
/*   var_j, var_k -- indices of the variables with respect to which the        */
/*                   2nd order partial derivative d^2[f(x)]/(dx_j)(dx_k)       */
/*                   is evaluated                                              */
/*   x            -- a point of the (GKLS_dim)-dimensional euclidean space     */
/* RETURN VALUE:                                                               */
/*   a second order partial derivative value OR                                */
/*   GKLS_MAX_VALUE if:(1) the index of a variable is out of range [1,GKLS_dim]*/
/*                     (2) the vector x does not belong to D;                  */
/*                     (3) the user tries to call the function without         */
/*                         parameter defining                                  */
/*******************************************************************************/
double GKLS_D2_deriv2(unsigned int var_j, unsigned int var_k, double *x)
{
 unsigned int i, index;
 double norm, scal, a, rho,
	    dh, difj, difk, hj, hk, dQ_jk; /* working variables */
 int the_same;  /* is TRUE if var_j==var_k */

 if ( (var_j == 0) || (var_j > GKLS_dim) ) return GKLS_MAX_VALUE;
 if ( (var_k == 0) || (var_k > GKLS_dim) ) return GKLS_MAX_VALUE;
 the_same = (var_j == var_k);
 var_j = var_j - 1; var_k = var_k - 1; /* to be used as indexes of array */

 if (!isArgSet) return GKLS_MAX_VALUE;

 for (i=0; i<GKLS_dim; i++)
  if ( (x[i] < GKLS_domain_left[i]-GKLS_PRECISION) || (x[i] > GKLS_domain_right[i]+GKLS_PRECISION) )
    return GKLS_MAX_VALUE;
 /* Check wether x belongs to some basin of local minima, M(index) <> T */
 /* Attention: number of local minima must be >= 2 */
 index = 1;
 while ((index<GKLS_num_minima) &&
        (GKLS_norm(GKLS_minima.local_min[index],x) > GKLS_minima.rho[index]) )
  index++;
 if (index == GKLS_num_minima)
 {
   /* Return the value of the second order partial derivative of the paraboloid function */
   if (the_same) return 2.0;
   else return 0.0;
 }

 /* Check wether x coincides with the local minimizer M(index) */
 if ( GKLS_norm(x, GKLS_minima.local_min[index]) < GKLS_PRECISION )
 {
   if (the_same) return delta;
   else return 0.0;
 }

 norm = GKLS_norm(GKLS_minima.local_min[0],GKLS_minima.local_min[index]);
 a = norm * norm + GKLS_minima.f[0] - GKLS_minima.f[index];
 rho = GKLS_minima.rho[index];
 norm = GKLS_norm(GKLS_minima.local_min[index],x);
 scal = 0.0;
 for(i=0; i<GKLS_dim; i++)
   scal += (x[i] - GKLS_minima.local_min[index][i]) *
           (GKLS_minima.local_min[0][i] - GKLS_minima.local_min[index][i]);
 difj = x[var_j] - GKLS_minima.local_min[index][var_j];
 difk = x[var_k] - GKLS_minima.local_min[index][var_k];
 hj = (GKLS_minima.local_min[0][var_j]-GKLS_minima.local_min[index][var_j])*norm -
     scal*difj/norm;
 hk = (GKLS_minima.local_min[0][var_k]-GKLS_minima.local_min[index][var_k])*norm -
     scal*difk/norm;

 dh = (GKLS_minima.local_min[0][var_j]-GKLS_minima.local_min[index][var_j])*difk/norm -
	   hk*difj/norm/norm;
 if (the_same) dh = dh - scal/norm;

 dQ_jk = -6.0/rho/rho/rho/rho*(dh*norm*norm*norm+3.0*hj*difk*norm) -
	     30.0/rho/rho/rho/rho*hk*difj*norm +
		 15.0/rho/rho/rho*(-6.0/rho*scal/norm+6.0/rho/rho*a+1-0.5*delta)*difj*difk*norm +
		 16.0/rho/rho/rho*(dh*norm*norm+2.0*hj*difk) +
		 64.0/rho/rho/rho*hk*difj +
		 8.0/rho/rho*(16.0/rho*scal/norm-15.0/rho/rho*a-3.0+1.5*delta)*difj*difk -
		 12.0/rho/rho*(dh*norm+hj*difk/norm) -
 		 36.0/rho/rho*hk*difj/norm +
         3.0/rho*(-12.0/rho*scal/norm+10.0/rho/rho*a+3.0-1.5*delta)*difj*difk/norm;

 if (the_same)
	 dQ_jk = dQ_jk +
	     5.0*norm*norm*norm/rho/rho/rho*(-6.0/rho*scal/norm+6.0/rho/rho*a+1-0.5*delta) +
         4.0*norm*norm/rho/rho*(16.0/rho*scal/norm-15.0/rho/rho*a-3.0+1.5*delta) +
		 3.0*norm/rho*(-12.0/rho*scal/norm+10.0/rho/rho*a+3.0-1.5*delta) +
		 delta;
 /* Return the value of d^2[Q(x)]/dx[var_j]dx[var_k] of the D2-type function */
 return dQ_jk;


} /* GKLS_D2_deriv2() */

/*******************************************************************************/
/*  The subroutine evaluates the gradient of the D-type test function          */
/* INPUT PARAMETERS:                                                           */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space                */
/*   g -- a pointer to the allocated array of the dimension GKLS_dim           */
/* RETURN VALUES:                                                              */
/*   an error code (that can be: GKLS_OK -- no error, or                       */
/*                               GKLS_DERIV_EVAL_ERROR -- otherwise)           */
/*   g -- a pointer to the array of gradient coordinates                       */
/*******************************************************************************/
int GKLS_D_gradient (double *x, double *g)
{
  unsigned int i;
  int error_code = GKLS_OK;

  if (!isArgSet) return GKLS_DERIV_EVAL_ERROR;
  if (g == NULL) return GKLS_DERIV_EVAL_ERROR;

  for (i=1; i<=GKLS_dim; i++)
  {
    g[i-1] = GKLS_D_deriv(i,x);
    if (g[i-1] >= GKLS_MAX_VALUE-1000.0)
		error_code = GKLS_DERIV_EVAL_ERROR;
  }
return error_code;
} /* GKLS_D_gradient() */


/*******************************************************************************/
/*  The subroutine evaluates the gradient of the D2-type test function         */
/* INPUT PARAMETERS:                                                           */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space                */
/*   g -- a pointer to the allocated array of the dimension GKLS_dim           */
/* RETURN VALUES:                                                              */
/*   an error code (that can be: GKLS_OK -- no error, or                       */
/*                               GKLS_DERIV_EVAL_ERROR -- otherwise)           */
/*   g -- a pointer to the array of gradient coordinates                       */
/*******************************************************************************/
int GKLS_D2_gradient (double *x, double *g)
{
  unsigned int i;
  int error_code = GKLS_OK;

  if (!isArgSet) return GKLS_DERIV_EVAL_ERROR;
  if (g == NULL) return GKLS_DERIV_EVAL_ERROR;

  for (i=1; i<=GKLS_dim; i++)
  {
    g[i-1] = GKLS_D2_deriv1(i,x);
    if (g[i-1] >= GKLS_MAX_VALUE-1000.0)
		error_code = GKLS_DERIV_EVAL_ERROR;
  }
return error_code;
} /* GKLS_D2_gradient() */


/*******************************************************************************/
/*  The subroutine evaluates the Hessian matrix of the D2-type test function   */
/* INPUT PARAMETERS:                                                           */
/*   x -- a point of the (GKLS_dim)-dimensional euclidean space                */
/*   h -- a pointer to the allocated matrix of dimension[GKLS_dim,GKLS_dim]    */
/* RETURN VALUES:                                                              */
/*   an error code (that can be: GKLS_OK -- no error, or                       */
/*                               GKLS_DERIV_EVAL_ERROR -- otherwise            */
/*   h -- a pointer to the Hessian matrix                                      */
/*******************************************************************************/
int GKLS_D2_hessian (double *x, double **h)
{
 unsigned int i, j;
 int error_code = GKLS_OK;

 if (!isArgSet) return GKLS_DERIV_EVAL_ERROR;
 if (h == NULL) return GKLS_DERIV_EVAL_ERROR;
 for (i=1; i<=GKLS_dim; i++)
   if (h[i-1] == NULL) return GKLS_DERIV_EVAL_ERROR;

 for (i=1; i<=GKLS_dim; i++)
	 for (j=1; j<=GKLS_dim; j++)
	 {
	   h[i-1][j-1] = GKLS_D2_deriv2(i,j,x);
       if (h[i-1][j-1] >= GKLS_MAX_VALUE-1000.0)
		error_code = GKLS_DERIV_EVAL_ERROR;
	 }
return error_code;
} /* GKLS_D2_hessian() */


/****************************** gkls.c ******************************/
