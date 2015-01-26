#PROJECTNAME = 

#.PHONY: default
#default: build

PYTHON = python
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print( sysconfig.get_python_inc() )'}
NUMPY_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print( numpy.get_include() )'}

TIMEIT = ${shell ${PYTHON} -c 'import timeit; print(timeit.__file__)'}

CYTHON = cython


SRC := $(wildcard *.pyx)
OBJ := $(addsuffix .o, $(basename $(SRC)))
CSRC := $(addsuffix .c, $(basename $(SRC)))
SOFILES := $(addsuffix .so, $(basename $(SRC)))

all: main

main: $(SRC)
	${PYTHON} setup.py build_ext --inplace

clean:
	-rm -f ${OBJ} ${CSRC}

