#PROJECTNAME = 

FC=gfortran
F2PY=f2py
F2PYFLAGS=--opt="-O3" --fcompiler="gnu95" --debug

FFLAGS=-Ofast

#SRC := $(wildcard *.f90)
SRC = \
      mod_linkedindex.f90 \
      ftnecf.f90 \
      ftnbp11.f90

SOSRC = \
      ftnecf.f90 \
      ftnbp11.f90

OBJ := $(addsuffix .o, $(basename $(SRC)))
SOFILES := $(addsuffix .so, $(basename $(SOSRC)))

all: main

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

ftnbp11.so: ftnecf.f90 ftnbp11.f90 mod_linkedindex.o
	${F2PY} -c ${F2PYFLAGS} -m ftnbp11 $^
	#${F2PY} -c ${F2PYFLAGS} -m ftnbp11 ftnecf.f90 ftnbp11.f90 only: initializelist addindextolist removelastitem getcurrentindex : skip : node indexqueue :

ftnecf.so: ftnecf.f90
	${F2PY} -c ${F2PYFLAGS} -m ftnecf $^

main: $(OBJ) $(SOFILES)

clean:
	-rm -f ./*.o ./*.mod ./*.so

