# Set the compiler and flags here. Note that the OpenMP option must be set.
# Uncomment below to use the GNU compiler.
F90 = gfortran -std=f2008 -fopenmp
F77 = gfortran -std=legacy -fopenmp
# Uncomment below to use the Intel compiler.
#F90 = ifort -std08 -qopenmp
#F77 = ifort -qopenmp
# Set the build flags here
COMP = -c
# List of object files
OBJS = vtmop_libe.o vtmop.o linear_shepard.o shared_modules.o \
       delsparse.o slatec.o qnstop.o lapack.o blas.o

# Build and test the generator and initializer functions for libEnsemble
all : $(OBJS)

# libE interface
vtmop_libe.o : vtmop_libe.f90 vtmop.o
	$(F90) $(COMP) vtmop_libe.f90 -o vtmop_libe.o


# Main VTMOP library
vtmop.o : vtmop.f90 delsparse.o linear_shepard.o qnstop.o
	$(F90) $(COMP) vtmop.f90 -o vtmop.o

# delsparse library, used to generate Delaunay graph
delsparse.o : delsparse.f90 shared_modules.o
	$(F90) $(COMP) delsparse.f90 -o delsparse.o

# linear Shepard's module
linear_shepard.o : linear_shepard.f90 shared_modules.o
	$(F90) $(COMP) linear_shepard.f90 -o linear_shepard.o

# QNSTOP contains a latin hypercube function
qnstop.o : qnstop.f90 shared_modules.o
	$(F90) $(COMP) qnstop.f90 -o qnstop.o

# real_precision module
shared_modules.o : shared_modules.f90
	$(F90) $(COMP) shared_modules.f90 -o shared_modules.o

# Subset of the slatec library, as needed for solving QPs
slatec.o : slatec.f
	$(F77) $(COMP) slatec.f -o slatec.o

# Subset of LAPACK library, as needed by VTMOP
lapack.o : lapack.f
	$(F77) $(COMP) lapack.f -o lapack.o

# Subset of BLAS library, as needed by VTMOP
blas.o : blas.f
	$(F77) $(COMP) blas.f -o blas.o

# Clean command
clean :
	rm -f *.o *.mod *.so
