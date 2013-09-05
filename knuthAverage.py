#!/usr/bin/env python
import numpy as np
#import pdb

def knuthUpdate(newValues,runningMean,runningVariance,runningCount):
  """ 
  
knuthUpdate(newValues,runningMean,runningVariance,runningCount)
      Do an online update of the mean and variance, following the Knuth
      algorithm.  See:
      http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

      Parameters
      ----------

      newValues : array_like
                  Array containing data to be added to the running average.  If
                  `a' is not an array, conversion is attempted.
      runningMean : float
                  Scalar float containing the running average of the data.
      runningVariance : float
                  Scalar float containing the running variance of the data.
      runningCount : float
                  Scalar float containing the running count of the data.

      Returns
      -------
      [newRunningMean, newRunningVariance, newRunningCount] : {float, float, float}
          The updated running mean, variance, and count for the data.

      Examples
      --------
  a = [1.0,2.0,3.0,4.0]
  b = [5.0,6.0,7.0,8.0]
  ave = np.average(a)
  var = np.var(a)
  count = len(a)

  print knuthUpdate(b,ave,var,count)
  a += b
  print np.average(a),np.var(a),len(a)

#[4.5, 5.25, 8]
#4.5 5.25 8

"""

  #Determine length of the new set
  newCount = len(np.asarray(newValues))

  #If there are no new points, just return the original values
  if(newCount <= 0.0):
    return [runningMean,runningVariance,runningCount]
  else:
    #Determine average of the new set
    newMean = np.average(newValues)
    
    #Calculate the variance of the new set
    newVar = np.var(newValues)

    #Return
    return knuthCombine(newMean,newVar,newCount,runningMean,runningVariance,runningCount)

def knuthCombine(meanA,varA,countA,meanB,varB,countB):
  """ 

knuthCombine(meanA,varA,countA,meanB,varB,countB)

      Do an online update of the mean and variance, following the Knuth
      algorithm.  See:
      http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

      Parameters
      ----------

      meanA, varA, countA : scalar
                  Scalar float containing the running mean, variance, and 
                  count for one portion of the sample

      meanB, varB, countB : scalar
                  Scalar float containing the running mean, variance, and 
                  count for another portion of the sample
                          
      Returns
      -------
      [newRunningMean, newRunningVariance, newRunningCount] : {float, float, float}
          The updated running mean, variance, and count for the data.

      Examples
      --------

"""

  #If there are no new points, just return the original values
  if(countA <= 0.0):
    return [meanB,varB,countB]
  if(countB <= 0.0):
    return [meanA,varA,countA]

  #Calculate the 2nd moments of the new and old sets
  M2a = varA*countA
  M2b = varB*countB

  #Calculate the length of the updated set
  newRunningCount = countA + countB

  #Calculate the delta parameter used in the update
  delta = meanA - meanB

  #Update the running mean
  newRunningMean = meanB + delta*countA/newRunningCount

  #Update the running 2nd moment
  newRunningM2 = M2b + M2a + (delta**2)*(countA*countB)/newRunningCount

  #Normalize the 2nd moment to be the variance
  newRunningVar = newRunningM2/newRunningCount

  #Return
  return [newRunningMean,newRunningVar,newRunningCount]
  


if(__name__ == "__main__"):

  ny = 50
  nx = 50
  randomNumbers = [ np.random.randn(nx) for i in range(ny) ]

  count = nx*ny

  randKnAverage = 0.0
  randKnVariance = 0.0
  randKnCount = 0.0

  for randKnVect in randomNumbers:
    [randKnAverage,randKnVariance,randKnCount] = \
      knuthUpdate(randKnVect,randKnAverage,randKnVariance,randKnCount)

  randAverage = np.average(randomNumbers)
  randVariance = np.var(randomNumbers)
  randCount = np.prod(np.shape(randomNumbers))

  print "Normal Average: mean = {} +/- {}, count = {}".format(randAverage, \
                                                              randVariance, \
                                                              randCount)
  print "Knuth Average: mean = {} +/- {}, count = {}".format( randKnAverage, \
                                                              randKnVariance, \
                                                              randKnCount)
  
