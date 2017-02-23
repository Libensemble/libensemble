/************************************************************************/
/*      The header file for linkage to the file rnd_gen.c               */
/************************************************************************/
#if !defined ( __RND_GEN_H )
#define __RND_GEN_H

#define KK 100                     /* the long lag */
#define LL  37                     /* the short lag */
#define mod_sum(x,y) (((x)+(y))-(int)((x)+(y)))   /* (x+y) mod 1.0 */

#define TT  70   /* guaranteed separation between streams */
#define is_odd(s) ((s)&1)

#define QUALITY 1009 /* recommended quality level for high-res use */

#define NUM_RND 1009 /* size of the array of random numbers */


extern double rnd_num[NUM_RND]; /* array of random numbers */

/* For rnd_gen.c */
void ranf_array(double aa[], int n); /* put n new random fractions in aa */
  /* double *aa  - destination */
  /* int n       - array length (must be at least KK) */
void ranf_start(long seed);  /* do this before using ranf_array */
  /* long seed   - selector for different streams */

#endif /* __RND_GEN_H */
