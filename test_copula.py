from numpy import *
import selfConsistentDensityEstimate as sc
import pylab as P

def covMat(sx,sy,r):
      return [[sx**2,r*sx*sy],[r*sx*sy,sy**2]]

size = 2**17


data = random.multivariate_normal([0.0,0.0],covMat(1.0,1.0,0.9),size).transpose()

scobj = sc.selfConsistentDensityEstimate(data)

xx,yy = meshgrid(*scobj.getTransformedAxes())
pdf = scobj.getTransformedPDF()
copdf = scobj.getTransformedCopula()

P.subplot(121)
P.contourf(xx,yy,pdf)
P.subplot(122)
P.contourf(xx,yy,copdf)
P.show()
