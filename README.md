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
'''
#!python
 
import numpy as np
import selfConsistentDensityEstimate as sc
import pylab as P
  
#Generate a 2x100000 random dataset (representing 100000 pairs of datapoints)
randomdata = np.reshape(np.random.normal(size=[200000]),[2,100000])

#Do the self-consistent density estimate
mysc = sc.selfConsistentDensityEstimate(randomdata)

#Get the axis values
x, y= mysc.getTransformedAxes()

#Get the PDF
myPDF2D = mysc.getTransformedPDF()

#Mask distribution indices that have less than 1 kernel contribution
badInds = mysc.findBadDistributionInds()
#(note that this is for presentation purposes; it unnormalizes the PDF)
myPDF2D[badInds] = 0.0

#Convert the x/y values to 2D grids for contouring
x2d,y2d = np.meshgrid(x,y)

#Plot contours of the PDF (should be a set of concentric circles)
P.contour(x2d,y2d,myPDF2D)
P.show()

'''


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

