# fastKDE #

## Software Overview ##

Calculates a self-consistent probability density estimate of arbitrarily
dimensioned data. By default, this imlementation uses a multidimensional
version of the nuFFT-based Empirical Characteristic Function calculation
described by O'Brien et al. (2014; Computational Statistics and Data Analysis,
doi:10.1016/j.csda.2014.06.002), which improves the speed of the
self-consistent density esimate described by Bernacchia and Pigolotti (2011; J.
Royal Statistical Society C, doi:10.1111/j.1467-9868.2011.00772.x)

Example usage:

###For a standard PDF

```
#!python
 
import numpy as np
import fastKDE
import pylab as P

#Generate two random variables dataset (representing 100000 pairs of datapoints)
N = 2e5
var1 = 50*np.random.normal(size=N) + 0.1
var2 = 0.01*np.random.normal(size=N) - 300
  
#Do the self-consistent density estimate
myPDF,axes = fastKDE.pdf(var1,var2)

#Extract the axes from the axis list
v1,v2 = axes

#Plot contours of the PDF should be a set of concentric ellipsoids centered on
#(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
#should be large
P.contour(v1,v2,myPDF)
P.show()

```

###For a conditional PDF

The following code generates samples from a non-trivial joint distribution
```python
from numpy import *

#***************************
# Generate random samples
#***************************
# Stochastically sample from the function underlyingFunction() (a sigmoid):
# sample the absicissa values from a gamma distribution
# relate the ordinate values to the sample absicissa values and add
# noise from a normal distribution

#Set the number of samples
numSamples = int(1e6)

#Define a sigmoid function
def underlyingFunction(x,x0=305,y0=200,yrange=4):
     return (yrange/2)*tanh(x-x0) + y0

xp1,xp2,xmid = 5,2,305  #Set gamma distribution parameters
yp1,yp2 = 0,12          #Set normal distribution parameters (mean and std)

#Generate random samples of X from the gamma distribution
x = -(random.gamma(xp1,xp2,int(numSamples))-xp1*xp2) + xmid
#Generate random samples of y from x and add normally distributed noise
y = underlyingFunction(x) + random.normal(loc=yp1,scale=yp2,size=numSamples)
```

**Now that we have the x,y samples, the following code calcuates the conditional**
```python
#***************************
# Calculate the conditional
#***************************
pOfYGivenX,axes = fastKDE.conditional(y,x)
```

The following plot shows the results:
```python
#***************************
# Plot the conditional
#***************************
fig,axs = PP.subplots(1,2,figsize=(10,5))

#Plot a scatter plot of the incoming data
axs[0].plot(x,y,'k.',alpha=0.1)
axs[0].set_title('Original (x,y) data')

#Set axis labels
for i in (0,1):
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

#Draw a contour plot of the conditional
axs[1].contourf(axes[0],axes[1],pOfYGivenX,64)
#Overplot the original underlying relationship
axs[1].plot(axes[0],underlyingFunction(axes[0]),linewidth=3,linestyle='--',alpha=0.5)
axs[1].set_title('P(y|x)')

#Set axis limits to be the same
xlim = [amin(axes[0]),amax(axes[0])]
ylim = [amin(axes[1]),amax(axes[1])]
axs[1].set_xlim(xlim)
axs[1].set_ylim(ylim)
axs[0].set_xlim(xlim)
axs[0].set_ylim(ylim)

fig.tight_layout()

PP.savefig('conditional_demo.png')
PP.show()
```
![Conditional PDF](conditional_demo.png)

## How do I get set up? ##

A standard python build:
```python setup.py install```

### Download the source ###

Please contact Travis A. O'Brien <TAOBrien@lbl.gov> to obtain the latest version of the source.

### Install pre-requisites ###
This code requires the following software:
  
  * Python >= 2.7.3
  * Numpy  >= 1.7
  * scipy
  * cython