#PROJECTNAME = 

FC=gfortran
F2PY=f2py
F2PYFLAGS=--opt="-O3" --fcompiler="gnu95"

FFLAGS=-Ofast

#SRC := $(wildcard *.f90)
SRC = \
      ftnbp11helper.f90

SOSRC = \
      ftnbp11helper.f90

OBJ := $(addsuffix .o, $(basename $(SRC)))
SOFILES := $(addsuffix .so, $(basename $(SOSRC)))

all: main

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

ftnbp11helper.so: ftnbp11helper.f90
	${F2PY} -c ${F2PYFLAGS} -m ftnbp11helper $^

main: $(OBJ) $(SOFILES)

clean:
	-rm -f ./*.o ./*.mod ./*.so

