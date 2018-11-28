import adaboost
from numpy import *
D = mat(ones((5,1))/5)
datMat,classLabels = adaboost.loadSimpData()
adaboost.buildStump(datMat,classLabels,D)