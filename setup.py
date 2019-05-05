from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os

inc_dirs = []
inc_dirs.append(numpy.get_include())
lib_dirs = []
# lib_dirs.append(numpy.get_lib())
libs = []

cmdclass = {'build_ext': build_ext}

# set commands to build the Cython assistance modules
extensions = [
    Extension("fastkde.floodFillSearch", ["fastkde/floodFillSearch.pyx"],
              libraries=libs,
              library_dirs=lib_dirs,
              include_dirs=inc_dirs,
              runtime_library_dirs=lib_dirs),
    Extension("fastkde.nufft", ["fastkde/nufft.pyx"],
              libraries=libs,
              library_dirs=lib_dirs,
              include_dirs=inc_dirs,
              runtime_library_dirs=lib_dirs),
]

# get revision information
with open('REVISION', 'r') as fin:
    revision = fin.read().rstrip()

# read the long description
with open('README', 'r') as fin:
    long_description = fin.read()

setup(
            name = 'fastkde',
            packages = ['fastkde'],
            version = revision,
            description = 'Tools for fast and robust univariate and multivariate kernel density estimation',
            long_description = long_description,
            author = "Travis A. O'Brien",
            author_email = "TAOBrien@lbl.gov",
            url = "https://bitbucket.org/lbl-cascade/fastkde",
            download_url = "https://bitbucket.org/lbl-cascade/fastkde/get/v{}.tar.gz".format(revision),
            keywords = ['statistics','probability','KDE','kernel density estimation'],
            py_modules=['fastkde.fastKDE','fastkde.empiricalCharacteristicFunction', 'fastkde.plot'],
            classifiers = [],
            cmdclass = cmdclass,
            ext_modules = extensions,
            )
