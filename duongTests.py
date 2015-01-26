#!/usr/bin/env python

#*******************************************************************************
#*******************************************************************************
#********************* Code importing ******************************************
#*******************************************************************************
#*******************************************************************************
from __future__ import division #Use Python 3 syntax for division (division of 
                                # integers results in a float)

from numpy import *
from scipy.stats import multivariate_normal
#(this requiquires scipy > 0.14 for the multivariate_normal function)

#plotting library
import pylab as P

#library for ordered dictionaries
import collections

#*******************************************************************************
#*******************************************************************************
#******************* Function and class definitions ****************************
#*******************************************************************************
#*******************************************************************************

#************************************
# Define a class for the Duong tests
#************************************

class sumOfNormals:

    def __init__(self,meanTuple,covMatTuple,fractionTuple):

        #Save the lists of means, covariance matricies, and fractional contributions
        self.means = meanTuple
        self.covMats = covMatTuple
        self.fractions = fractionTuple

    def __call__(self,x,y):
        #Predeclare the return value
        returnVal = zeros(shape(x)) 

        #Concatenate x/y for input into multivariate_normal()
        xy = concatenate((asarray(x)[...,newaxis],asarray(y)[...,newaxis]), \
                            axis=-1)

        #Loop over the list of normal distribution paramters and add the
        #fractional contribution from each normal at the x/y points
        for mu,cov,frac in zip(self.means,self.covMats,self.fractions):
            returnVal += frac*multivariate_normal.pdf(xy,mean=mu,cov=cov)

        return returnVal

    def getSample(self,size=1):
        """Pull samples from the summed normal distribution, assuming that the
        number of sample sizes is large such that roughly f*N samples are drawn
        for each normal component, where f is the fraction of that normal
        component."""

        sizes = []
        #Loop through all but the last normal and calculate the number
        #of samples to draw, based on the fraction of that normal
        for frac in self.fractions[:-1]:
           sizes.append(int(round(frac*size))) 

        #Calculate the size of the last sample to make sure that we return a total
        #sample of size 'size'; this assumes that sum(self.fractions) = 1
        sizes.append(int(size - sum(sizes)))

        #Generate samples for each normal
        randSamples = []
        for mu,cov,frac,N in zip(self.means,self.covMats,self.fractions,sizes):
            randSamples.append(transpose(multivariate_normal.rvs(mean=mu,cov=cov,size=N)))

        #Concatenate the gaussian samples
        randSample = concatenate(tuple(randSamples),axis=1)


        #Shuffle the samples along the long axis so that we
        #can draw successively larger samples
        ishuffle = asarray(range(size))
        random.shuffle(ishuffle)
        randSample = randSample[:,ishuffle]

        return randSample

        
        

#*******************************************************************************
#*******************************************************************************
#********************** Generate each of the Duong tests ***********************
#*******************************************************************************
#*******************************************************************************

#predeclare a dict for the duongTestDict
duongTestDict = collections.OrderedDict()

#***********
# Test A
#***********
means = ([0,0],)
covMats = ( \
            [ [1/4, 0], \
              [0,   1] ], )
fractions = (1.0,)
duongTestDict['A'] = sumOfNormals(means,covMats,fractions)

#***********
# Test B
#***********
means = ([1,0], [-1,0])
covMats = ( \
            [ [4/9, 0], \
              [0,   4/9] ], \
            [ [4/9, 0], \
              [0,   4/9] ], \
          )
fractions = (1/2,  1/2)
duongTestDict['B'] = sumOfNormals(means,covMats,fractions)

#***********
# Test C
#***********
means = ([3/2,0], [-3/2,0])
covMats = ( \
            [ [1/16,    0], \
              [0,       1] ], \
            [ [1/16,    0], \
              [0,       1] ], \
          )
fractions = (1/2,  1/2)
duongTestDict['C'] = sumOfNormals(means,covMats,fractions)

#***********
# Test D
#***********
means = ([1,-1], [-1,1])
covMats = ( \
            [ [4/9,     14/45   ], \
              [14/45,   4/9     ] ], \
            [ [4/9,     0], \
              [0,       4/9     ] ], \
          )
fractions = (1/2,  1/2)
duongTestDict['D'] = sumOfNormals(means,covMats,fractions)

#***********
# Test E
#***********
means = ([-1,0],[1, 2/sqrt(3)], [1, -2/sqrt(3)])
covMats = ( \
            [ [9/25,    63/250  ], \
              [63/250,  49/100  ] ], \
            [ [9/25,      0     ], \
              [ 0    ,  49/100  ] ], \
            [ [9/25,      0     ], \
              [ 0    ,  49/100  ] ], \
          )
fractions = (3/7,  3/7, 1/7)
duongTestDict['E'] = sumOfNormals(means,covMats,fractions)

#***********
# Test F
#***********
means = ([0,0],)
covMats = ( \
            [ [1,       9/10], \
              [9/10,    1,] ], )
fractions = (1,)
duongTestDict['F'] = sumOfNormals(means,covMats,fractions)


#Seed with 0 so results are reproducable
#random.seed(0)

#*******************************************************************************
#*******************************************************************************
#***************** Run unit test ***********************************************
#*******************************************************************************
#*******************************************************************************
if(__name__ == "__main__"):
    #Generate a plot comparable to Figure 2.1 in Duong (2005) to verify
    #that the Duong tests all work as expected.  Also draw samples
    #from the constructed distributions to demonstrate that the samples
    #lay approximately where expected

    #Generate x/y coordinates for the plots
    xtmp = linspace(-3,3,100)
    x2d,y2d = meshgrid(xtmp,xtmp)

    #Loop over the Duong test objects
    i = 1
    for key,testObj in duongTestDict.iteritems():
        #Plot a contour of the PDF
        P.subplot(3,2,i)
        P.contour(x2d,y2d,duongTestDict[key](x2d,y2d))

        #Draw samples and plot slighly transparent points for each sample
        samples = testObj.getSample(1000)
        xs = samples[0,:]
        ys = samples[1,:]
        P.plot(xs,ys,'.',color=[0,0,0,0.3],markersize=2)

        #Add a title marking which test
        P.title('Target density {}'.format(key))

        #increment the panel counter
        i += 1

    #Spread the plots apart
    P.tight_layout()
    #Draw the plot
    P.show()







