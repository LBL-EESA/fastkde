# selfConsistentDensityEstimate #

## Software Overview ##

Calculates a self-consistent probability density estimate of arbitrarily
dimensioned data. By default, this imlementation uses a multidimensional
version of the nuFFT-based Empirical Characteristic Function calculation
described by O'Brien et al. (2014; Computational Statistics and Data Analysis,
doi:10.1016/j.csda.2014.06.002), which improves the speed of the
self-consistent density esimate described by Bernacchia and Pigolotti (2011; J.
Royal Statistical Society C, doi:10.1111/j.1467-9868.2011.00772.x)

Example usage:

```
#!python
 
import numpy as np
import selfConsistentDensityEstimate as sc
import pylab as P

#Generate two random variables dataset (representing 100000 pairs of datapoints)
N = 2e5
var1 = 50*np.random.normal(size=N) + 0.1
var2 = 0.01*np.random.normal(size=N) - 300
  
#Do the self-consistent density estimate
myPDF,axes = sc.pdf(var1,var2)

#Extract the axes from the axis list
v1,v2 = axes

#Plot contours of the PDF should be a set of concentric ellipsoids centered on
#(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
#should be large
P.contour(v1,v2,myPDF)
P.show()

```


## How do I get set up? ##

In the present code version, setup consists of (1) downloading the source, (2) building fortran sub-modules, and (3) setting the PYTHONPATH.

### Download the source ###

Please contact Travis A. O'Brien <TAOBrien@lbl.gov> to obtain the latest version of the source.

### Install pre-requisites ###
This code requires the following software:
  
  * Python >= 2.7.3
  * Numpy  >= 1.7
  * f2py (should be distributed with numpy)
  * GNU make
  

### Building Fortran sub-modules ###

Prior to use, some Fortran-based Python extensions must be built.  Do the
following from the command line (assuming that the current working directory is
the base of the repository):

```
#!bash
     make
```

### Set the PYTHONPATH ###

Do the following from the command line (assuming that the current working
directory is the base of the repository):

(for Bourne shell)
```
#!bash
  export PYTHONPATH=${PYTHONPATH}:${PWD}
```

(for c-shell)
```
#!csh
  setenv PYTHONPATH ${PYTHONPATH}:${PWD}
```

For permanent installation, please consider adding similar lines to your
.bashrc/.bash\_profile or .cshrc files (replacing ${PWD} with the full path of
the selfConsistentDensityEstimate directory).