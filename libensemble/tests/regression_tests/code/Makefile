FORT = gfortran -fcheck=all -std=f2008
COMP = -c 
LIBS = -llapack -lblas
OBJS = mop.o linear_shepard.o delaunaygraph.o slatec.o

all : mop_gen mop_init
	mv mop_gen ../mop_gen
	mv mop_init ../mop_init

mop_gen : mop_gen.f90 $(OBJS)
	$(FORT) mop_gen.f90 $(OBJS) $(LIBS) -o mop_gen

mop_init : mop_init.f90 $(OBJS)
	$(FORT) mop_init.f90 $(OBJS) $(LIBS) -o mop_init

mop.o : mop.f08 linear_shepard.o delaunaygraph.o
	$(FORT) $(COMP) mop.f08 -o mop.o

linear_shepard.o : linear_shepard.f95 delaunaygraph.o
	$(FORT) $(COMP) linear_shepard.f95 -o linear_shepard.o

delaunaygraph.o : delaunaygraph.f90
	$(FORT) $(COMP) delaunaygraph.f90 -o delaunaygraph.o

slatec.o : slatec.f
	$(FORT) $(COMP) slatec.f -o slatec.o

clean :
	rm -f *.o *.mod test
