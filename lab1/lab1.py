import numpy
from numpy import *
import pylab
from pylab import * 
import mlp
import random
f = open('class1','r')

scale = double(raw_input("Scale for class2: "))
center = double(raw_input("Center for class2: "))
nhidden = int(raw_input("Number of hidden: "))
beta = int(raw_input("Beta: "))
momentum = double(raw_input("Momentum: "))
eta = double(raw_input("Eta: "))
iterations = int(raw_input("Iterations: "))

#CREATE CLASSES
class1 = numpy.random.randn(50,3)
for i in range(len(class1)):
    for j in range(3):
        class1[i,j] = double(f.readline())

print class1

class2 = numpy.random.randn(50,3)
class2 = class2*scale+center
class2[:,2] = 1

#FUSE CLASSES INTO ONE DATASET
dataset = numpy.concatenate((class1,class2),axis=0)
#SHUFFLE THE DATASET
order = range(numpy.shape(dataset)[0])
random.shuffle(order)
dataset = dataset[order,:]

#CReATE TRAINDATA, VALIDATIONDATA, TESTDATA
train = dataset[0:50,0:2]
traint = numpy.zeros((len(train),1))
indices = numpy.where(dataset[0:50,2]==1)
traint[indices,0] = 1

valid = dataset[50:75,0:2]
validt = numpy.zeros((len(valid),1))
indices = numpy.where(dataset[50:75,2]==1)
validt[indices,0] = 1

test = dataset[75:100,0:2]
testt = numpy.zeros((len(train),1))
indices = numpy.where(dataset[75:100,2]==1)
testt[indices,0] = 1;

#TRAINING!!!
ANN = mlp.mlp(train,traint,nhidden,beta,momentum)
ANN.mlptrain(train,traint,eta,iterations)
ANN.confmat(train,traint)

#PLOTTING?
xRange = numpy.arange(-4,4,0.1)                           ###
yRange = numpy.arange(-4,4,0.1)                             #
xgrid, ygrid = numpy.meshgrid(xRange,yRange)                #
                                                            #
noOfPoints = xgrid.shape[0] * xgrid.shape[1]                #
xcoords = xgrid.reshape((noOfPoints, 1))                    ### CAN BE DONE WITH 2 FORLOOPS
ycoords = ygrid.reshape((noOfPoints, 1))                    #
samples = numpy.concatenate((xcoords,ycoords), axis=1)      #
                                                            #
ones = -numpy.ones(xcoords.shape)                           #
samples = numpy.concatenate((samples,ones), axis=1)       ###

indicator = ANN.mlpfwd(samples)
indicator = indicator.reshape(xgrid.shape)

pylab.contour(xRange,yRange,indicator, (0.5,))
#pylab.plot(0, 1, 'o', 1, 0, 'o',1,1,'o',0,0,'o')
pylab.plot(class1[:,0], class1[:,1], 'o',class2[:,0], class2[:,1],'o')
pylab.show()
