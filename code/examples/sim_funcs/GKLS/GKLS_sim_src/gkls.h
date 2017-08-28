/******************************************************************************/
/*        GKLS-Generator of Classes of ND  (non-differentiable),              */
/*                                 D  (continuously differentiable), and      */
/*                                 D2 (twice continuously differentiable)     */
/*                     Test Functions for Global Optimization                 */
/*                                                                            */
/*   Authors:                                                                 */
/*                                                                            */
/*   M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                         */
/*                                                                            */
/*   (C) 2002-2005                                                            */
/*                                                                            */
/*	 References:                                                              */
/*                                                                            */
/*   1. M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev (2003),              */
/*   Algorithm 829: Software for Generation of Classes of Test Functions      */
/*   with Known Local and Global Minima for Global Optimization.              */
/*   ACM Transactions on Mathematical Software, Vol. 29, no. 4, pp. 469-480.  */
/*                                                                            */
/*   2. D.Knuth (1997), The Art of Computer Programming, Vol. 2:              */
/*   Seminumerical Algorithms (Third Edition). Reading, Massachusetts:        */
/*   Addison-Wesley.                                                          */
/*                                                                            */
/*   The software constructs a convex quadratic function (paraboloid) and then*/
/*   systematically distorts randomly selected parts of this function         */
/*   by polynomials in order to introduce local minima and to construct test  */
/*   functions which are non-differentiable (ND-type), continuously           */
/*   differentiable (D-type), and twice continuously differentiable (D2-type) */
/*   at the feasible region.                                                  */
/*                                                                            */
/*   Each test class is defined by the following parameters:                  */
/*  (1) problem dimension                                                     */
/*  (2) number of local minima including the paraboloid min and the global min*/
/*  (3) global minimum value                                                  */
/*  (3) distance from the paraboloid vertex to the global minimizer           */
/*  (4) radius of the attraction region of the global minimizer               */
/******************************************************************************/

#if !defined( __GKLS_H )
#define __GKLS_H

/* Penalty value of the generated function if x is not in D */
#define GKLS_MAX_VALUE        1E+100

/* Value of the machine zero in the floating-point arithmetic */
#define GKLS_PRECISION        1.0E-10

/* Default value of the paraboloid minimum */
#define GKLS_PARABOLOID_MIN   0.0

/* Global minimum value: to be less than GKLS_PARABOLOID_MIN */
#define GKLS_GLOBAL_MIN_VALUE -1.0

/* Max value of the parameter delta for the D2-type class function        */
/* The parameter delta is chosen randomly from (0, GKLS_DELTA_MAX_VALUE ) */
#define GKLS_DELTA_MAX_VALUE  10.0

/* Constant pi */
#ifndef PI
#define PI 3.14159265
#endif

/* Error codes */
#define GKLS_OK                              0
#define GKLS_DIM_ERROR                       1
#define GKLS_NUM_MINIMA_ERROR                2
#define GKLS_FUNC_NUMBER_ERROR               3
#define GKLS_BOUNDARY_ERROR                  4
#define GKLS_GLOBAL_MIN_VALUE_ERROR          5
#define GKLS_GLOBAL_DIST_ERROR               6
#define GKLS_GLOBAL_RADIUS_ERROR             7
#define GKLS_MEMORY_ERROR                    8
#define GKLS_DERIV_EVAL_ERROR                9

/* Reserved error codes */
#define GKLS_GREAT_DIM                      10
#define GKLS_RHO_ERROR                      11
#define GKLS_PEAK_ERROR                     12
#define GKLS_GLOBAL_BASIN_INTERSECTION      13

/* Internal error codes */
#define GKLS_PARABOLA_MIN_COINCIDENCE_ERROR 14
#define GKLS_LOCAL_MIN_COINCIDENCE_ERROR    15
#define GKLS_FLOATING_POINT_ERROR           16



/* The next two structures define a list of all local minima and    */
/* a list of the global minima.                                     */
/* These lists are filled by the generator.                         */
/* The fields of the structures help to the user                    */
/* to study properties of a concrete generated test function        */

/* The structure of type T_GKLS_Minima contains the following          */
/* information about all local minima (including the paraboloid        */
/* minimum and the global one): coordinates of local minimizers,       */
/* local minima values, and attraction regions radii.                  */
typedef struct {
         double **local_min; /* list of local minimizers coordinates   */
         double *f;          /* list of local minima values            */
		 double *w_rho;      /* list of radius weights                 */
         double *peak;       /* list of parameters gamma(i) =          */
		                     /*  = local minimum value(i) - paraboloid */
		                     /*    minimum within attraction regions   */
		                     /*    of local minimizer(i)               */
		 double *rho;        /* list of attraction regions radii       */
} T_GKLS_Minima;

/* The structure of type T_GKLS_GlobalMinima contains information      */
/* about the number of global minimizers and their                     */
/* indices in the set of local minimizers                              */

typedef struct {
	     unsigned int num_global_minima; /* number of global minima    */
		 unsigned int *gm_index;  /* list of indices of generated      */
		 /* minimizers, which are the global ones (elements from 0     */
		 /* to (num_global_minima - 1) of the list) and the local ones */
         /* (the resting elements of the list)                         */
} T_GKLS_GlobalMinima;


/*-------------- Variables accessible by the user --------------------- */
extern double *GKLS_domain_left; /* left boundary vector of D  */
  /* D=[GKLS_domain_left; GKLS_domain_ight] */
extern double *GKLS_domain_right;/* right boundary vector of D */

extern unsigned int GKLS_dim;    /* dimension of the problem,        */
                                 /* 2<=test_dim<NUM_RND (see random) */
extern unsigned int GKLS_num_minima; /* number of local minima, >=2  */

extern double GKLS_global_dist;  /* distance from the paraboloid minimizer  */
                                 /* to the global minimizer                 */
extern double GKLS_global_radius;/* radius of the global minimizer          */
                                 /* attraction region                       */
extern double GKLS_global_value; /* global minimum value,                   */
                                 /* test_global_value < GKLS_PARABOLOID_MIN */
extern T_GKLS_Minima GKLS_minima;
                                 /* see the structures type description     */
extern T_GKLS_GlobalMinima GKLS_glob;


/*------------------------User function prototypes -------------------------*/

int GKLS_domain_alloc(void); /* allocate boundary vectors   */

void GKLS_domain_free(void); /* deallocate boundary vectors */

int  GKLS_set_default(void); /* set default values of the input parameters  */
                          /* and allocate the boundary vectors if necessary */
void GKLS_free(void);        /* deallocate memory needed for the generator  */

int GKLS_parameters_check(void);/* test the validity of the input parameters*/

int GKLS_arg_generate (unsigned int); /* test function generator */

double GKLS_ND_func(double *);  /* evaluation of an ND-typed test function  */

double GKLS_D_func(double *);   /* evaluation of a D-typed test function    */

double GKLS_D2_func(double *);  /* evaluation of a D2-type test function    */

double GKLS_D_deriv(unsigned int, double *);
             /* first order partial derivative of the D-typed test function   */
double GKLS_D2_deriv1(unsigned int, double *);
             /* first order partial derivative of the D2-typed test function  */
double GKLS_D2_deriv2(unsigned int, unsigned int, double *);
             /* second order partial derivative of the D2-typed test function */
int GKLS_D_gradient  (double *, double *); /* gradient of the D-type test function  */

int GKLS_D2_gradient (double *, double *); /* gradient of the D2-type test function */

int GKLS_D2_hessian  (double *, double **);/* Hessian of the D2-type test function  */


#endif  /* __GKLS_H */
