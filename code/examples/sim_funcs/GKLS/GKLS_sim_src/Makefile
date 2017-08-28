# Example of a UNIX makefile provided to
# UNIX users for a simple compilation and linkage of separate files of
# the application project.
#
# Authors:                                                                 
#
# M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                     
#
CC = gcc
CFLAGS = -g -Wall

gkls : example.o gkls.o rnd_gen.o
	$(CC) $(CFLAGS) example.o gkls.o rnd_gen.o -o gkls -lm 
gkls_single : single_feval.o gkls.o rnd_gen.o
	$(CC) $(CFLAGS) single_feval.o gkls.o rnd_gen.o -o gkls_single -lm 
example.o : gkls.h rnd_gen.h example.c
	$(CC) -I. $(CFLAGS) -c example.c
single_feval.o : gkls.h rnd_gen.h single_feval.c
	$(CC) -I. $(CFLAGS) -c single_feval.c
gkls.o : gkls.h rnd_gen.h gkls.c
	$(CC) -I. $(CFLAGS) -c gkls.c
rnd_gen.o : rnd_gen.h rnd_gen.c
	$(CC) -I. $(CFLAGS) -c rnd_gen.c
clean:
	-rm *.o
