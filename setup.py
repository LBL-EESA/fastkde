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

extensions = [Extension("floodFillSearchC",["floodFillSearch.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]

setup(
            cmdclass = cmdclass,\
            ext_modules = extensions, \
            )
