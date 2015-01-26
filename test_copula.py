from numpy import *
import selfConsistentDensityEstimate as sc
from mpl_toolkits.mplot3d import axes3d
import pylab as P
import matplotlib as mpl

def covMat(sx,sy,r):
      return [[sx**2,r*sx*sy],[r*sx*sy,sy**2]]

size = 2**19


data = random.multivariate_normal([0.0,0.0],covMat(1.0,1.0,0.0),size).transpose()

scobj = sc.selfConsistentDensityEstimate(data,countThreshold = 100)

x,y = scobj.getTransformedAxes()
xx,yy = meshgrid(x,y)
pdf = scobj.getTransformedPDF()
copdf = scobj.getTransformedCopula()
print amin(copdf),amax(copdf)

norm = where(copdf == copdf,1.0,0.0)

fig = P.figure()
ax1 = fig.add_subplot(131)
ax1.contourf(xx,yy,pdf)
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
#ax2 = fig.add_subplot(122,projection='3d')
ax2 = fig.add_subplot(132)
#ax2.plot_surface(xx,yy,copdf,rstride = 2,cstride = 2)
ax2.contourf(xx,yy,copdf,norm=mpl.colors.LogNorm())
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])

ax3 = fig.add_subplot(133)
ax3.plot(x,sum(copdf,axis=1)/sum(norm,axis=1))
ax3.plot(y,sum(copdf,axis=0)/sum(norm,axis=0))
P.show()
