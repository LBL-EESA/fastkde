from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

inc_dirs = []
inc_dirs.append(numpy.get_include())
lib_dirs = []
#lib_dirs.append(numpy.get_lib())
libs = []

cmdclass = {'build_ext': build_ext}

extensions = [ \
    Extension("floodFillSearch",["floodFillSearch.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs),\
    Extension("nufft",["nufft.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs,extra_compile_args=["-fopenmp"],extra_link_args=['-fopenmp']),\
                ]

setup(
            cmdclass = cmdclass,\
            ext_modules = extensions, \
            )
