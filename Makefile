#Name of C compiler
CCMPL = gcc
#Name of Fortran compiler
FPATH = 
F90 = $(FPATH)ifort
F77 = $(FPATH)ifort
#compiling object file flags
OPT1 = -O3 -mkl -qopenmp -heap-arrays
#OPT1 = -O0 -g -CB -mkl
#OPT1 = -O3
OPT2 = 
FFLAGS = -c $(OPT1) $(OPT2)
#FFLAGS = -c -O0 -g -CB -warn $(OPT2)
#linking flags
LFLAGS = $(OPT1) $(OPT2)
#LFLAGS = -O0 -g -CB -warn $(OPT2)
FITSIODIR = /usr/local/lib
#Pgplot plot libraries
PGPLOTDIR = /usr/local/lib
#X11 libraries
X11DIR = /usr/X11/lib
# Libraries for linking pgplot
LIBS = -L$(PGPLOTDIR) -L$(X11DIR) -lX11 -lpgplot -lpng
# libraries for linking CFITSIO
LIBS2 = -L$(PGPLOTDIR) -L$(X11DIR) -L$(FITSIODIR) -lX11 -lpgplot -lcfitsio -lpng
#Directory where executable are placed
BIN = ../bin/
#utils source directory
UTILS = utils/

#Listing of programs to create.
all: neossatphot combinedark

neossatphotincl = precision.o readfits.o displayfits.o rqsort.o stdev.o heatlut.o \
 darkcorrect.o medfit.o columncor.o median.o openfilelist.o getstarlist.o \
 centroids.o
neossatphot: neossatphot.f90 $(neossatphotincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ $< $(neossatphotincl) $(LIBS2)

combinedarkincl = precision.o readfits.o displayfits.o rqsort.o stdev.o heatlut.o \
 combine.o openfilelist.o writefits.o deletefile.o
combinedark: combinedark.f90 $(combinedarkincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ $< $(combinedarkincl) $(LIBS2)

#building object libraries
%.o : $(UTILS)%.f90
	$(F90) $(FFLAGS) -o $@ $< 

%.o : $(UTILS)%.f
	$(F90) $(FFLAGS) -o $@ $< 

# Removing object files
.PHONY : clean
clean :
	rm *.o
	rm *.mod
