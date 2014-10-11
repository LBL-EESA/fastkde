#PROJECTNAME = 

#.PHONY: default
#default: build

PYTHON = python2.7
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print( sysconfig.get_python_inc() )'}
NUMPY_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print( numpy.get_include() )'}

TIMEIT = ${shell ${PYTHON} -c 'import timeit; print(timeit.__file__)'}

CYTHON = cython-2.7


FC=gfortran
F2PY=f2py
F2PYFLAGS=--opt="-O3 -g" --fcompiler="gnu95"
#F2PYFLAGS=--opt="-O0 -g" --fcompiler="gnu95"

FFLAGS=-O3 -g
#FFLAGS=-O0 -g

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

SOFILES := floodFillSearchC${SO}

SRC := floodFillSearch.c

all: main

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

ftnbp11.so: ftnecf.f90 ftnbp11.f90 mod_linkedindex.o
	${F2PY} -c ${F2PYFLAGS} -m ftnbp11 $^
	#${F2PY} -c ${F2PYFLAGS} -m ftnbp11 ftnecf.f90 ftnbp11.f90 only: initializelist addindextolist removelastitem getcurrentindex : skip : node indexqueue :

ftnecf.so: ftnecf.f90
	${F2PY} -c ${F2PYFLAGS} -m ftnecf $^

floodFillSearch.c: floodFillSearch.pyx
	${CYTHON} -I ${PYTHON_INCLUDE} -I${NUMPY_INCLUDE} $<

floodFillSearchC${SO}: floodFillSearch.pyx
	${PYTHON} setup.py build_ext --inplace

main: $(OBJ) $(SOFILES)

clean:
	-rm -f ./*.o ./*.mod ./*.so floodFillSearch.c

