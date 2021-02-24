from setuptools import setup,Extension
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
with open('README.rst', 'r') as fin:
    long_description = fin.read()

with open('requirements.txt', 'r') as fin:
    install_requires = fin.read().split()

extras = {
    "test": ["pytest"],
    "dev": [],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(
    name='fastkde',
    packages=['fastkde'],
    version=revision,
    description='Tools for fast and robust univariate and multivariate kernel density estimation',
    long_description=long_description,
    author="Travis A. O'Brien",
    author_email="obrienta@iu.edu",
    url="https://github.com/LBL-EESA/fastkde",
    download_url="https://github.com/LBL-EESA/fastkde/archive/v{}tar.gz".format(revision),
    keywords=['statistics', 'probability', 'KDE', 'kernel density estimation'],
    py_modules=['fastkde.fastKDE',
                'fastkde.empiricalCharacteristicFunction',
                'fastkde.plot'],
    classifiers=[],
    cmdclass=cmdclass,
    ext_modules=extensions,
    install_requires=install_requires,
    extras_require=extras
)
