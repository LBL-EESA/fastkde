from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os


class codeVersionID:

  def __init__(self,file=None):
    """ 
    A module to automatically obtain code version information if possible, and
    the julian time if not.

    Attempts to use 'hg id' to obtain version information.  If that fails, it
    attempts to read version information from a file called REVISION, and
    finally returns the current julian time if all else fails.
    """

    #By default assume that we are using mercurial
    self.reportingFromFile = False
    self.reportingTime = False

    if(file is not None):
      import os
      try:
        #Get the file's directory
        fileDir = os.path.dirname(os.path.realpath(file))
        #Assume that the repository is the same directory as this file
        repoDir = "/".join(fileDir.split('/')[:])
      except:
        repoDir = "./"

    try:
    #First, try to read the version information from mercurial
      import subprocess
      mercID=subprocess.check_output(["hg","id",repoDir])
      ID=mercID.split(' ')[0]
    except:
    #Second, try to read the version information from a REVISION file
    #(this would be used if we strip hg information from a tagged repo version
    # and store the version code in the REVISION file)
      try:
        with open("{}/REVISION".formate(repoDir),'r') as fin:
          contents = fin.read()
          ID=contents.split(' ')[0]
        self.reportingFromFile = True
      except:
      #Finally, at very worst, return the current timestamp so that 
      #version information might be inferred at a later date
        import time
        ID = int(time.time())
        self.reportingTime = True

    #Set the ID property
    self.ID = ID

    cwd = os.getcwd()
    try:
    #First try to read branch information from mercurial
      import subprocess
      import os
      os.chdir(repoDir)
      mercBranch = subprocess.check_output(["hg","branch"])
      os.chdir(cwd)
      branch = mercBranch.split(" ")[0].rstrip()
    except:
    #Otherwise, set it to nothing
      branch = None
    
    #Make sure that we end back up in the original directory
    os.chdir(cwd)

    #Set the branch property
    self.branch = branch

inc_dirs = []
inc_dirs.append(numpy.get_include())
lib_dirs = []
#lib_dirs.append(numpy.get_lib())
libs = []

cmdclass = {'build_ext': build_ext}

extensions = [ \
    Extension("floodFillSearch",["floodFillSearch.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs),\
    Extension("nufft",["nufft.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs),\
    #Extension("nufft",["nufft.pyx"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs,extra_compile_args=["-fopenmp"],extra_link_args=['-fopenmp']),\
                ]

setup(
            name = 'fastKDE', \
            version = str(codeVersionID(__file__).ID), \
            py_modules=['fastKDE','empiricalCharacteristicFunction'], \
            cmdclass = cmdclass,\
            ext_modules = extensions, \
            )
