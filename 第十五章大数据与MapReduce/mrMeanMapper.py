# 构建一个海量数据上分布式计算均值和方差的MapReduce
# 分布式均值和方差计算的mapper
import sys
# from numpy import mat,mean,power
from numpy import *
def read_input(file):
    for line in file:
        yield line.rstrip()
input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = powr(input,2)
print("%d\t%f\t%f" %(numInputs,mean(input),mean(sqInput)))
print(sys.stdin,"report: still alive")

# 分布式计算均值和方差的reducer
import sys
# from numpy import mat,mean,power
from numpy import *
def read_input(file):
    for line in file:
        yield line.rstrip()
input = read_input(sys.stdin)
mapperout = [line.split('\t') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperout:
    nj = float(instance)
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
mean = cumVal/cumN
varSum = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
print("%d\t%f\t%f" %(numInputs,mean(input),mean(sqInput)))
print(sys.stdin,"report: still alive")
