from fastkde import fastKDE
import pylab as PP
import numpy as np
import matplotlib as mpl

# set plot default fonts (fonts that are generally nice figures
font = {    'family' : 'serif', \
            'size' : '15', \
            'weight' : 'bold'}
mpl.rc('font', **font)
mpl.rc('axes', labelweight = 'bold') # needed for bold axis labels in more recent version of matplotlib


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
     return (yrange/2)*np.tanh(x-x0) + y0

xp1,xp2,xmid = 5,2,305  #Set gamma distribution parameters
yp1,yp2 = 0,12          #Set normal distribution parameters (mean and std)

#Generate random samples of X from the gamma distribution
x = -(np.random.gamma(xp1,xp2,int(numSamples))-xp1*xp2) + xmid
#Generate random samples of y from x and add normally distributed noise
y = underlyingFunction(x) + np.random.normal(loc=yp1,scale=yp2,size=numSamples)

#***************************
# Calculate the conditional
#***************************
pOfYGivenX,axes = fastKDE.conditional(y,x)

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
xlim = [np.amin(axes[0]),np.amax(axes[0])]
ylim = [np.amin(axes[1]),np.amax(axes[1])]
axs[1].set_xlim(xlim)
axs[1].set_ylim(ylim)
axs[0].set_xlim(xlim)
axs[0].set_ylim(ylim)

fig.tight_layout()

PP.show()
