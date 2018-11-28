# Logistic回归梯度上升优化算法
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        # 1行100列的行向量
    return dataMat,labelMat

from numpy import *
def sigmoid(inx):
    # return 1.0/(1+ exp(-inx))
    # 优化这个方法，inx<0时，exp(-inx)有可能越界
    if inx >= 0:
        return 1.0/(1 + exp(-inx))
    else:
        return exp(inx)/(1 + exp(inx))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    # transpose()表示numpy中的转置函数
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        # error表示算出的值与原始给定的之间的误差
        error = (labelMat - h)
        # 这一步和代价函数求法类似，代价函数的求得是通过方差来求得的，通过都代价函数进行求导，也就是对方差进行求导得到最终的更新函数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        # 不能将序列乘以非int 解决方式是将其转换为array数组就可以了
        weights = weights + alpha * error * array(dataMatrix[i])
    return weights

import random
# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        # dataInx = range(m)采用在这种方式无法在后面对其进行删除
        dataInx = list(range(m))
        for i in range(m):
            # alpha 每次迭代时需要进行不断的调整
            # alpha 在不断减少，但是绝不可能减小为0，因为有一个常数项0.01
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataInx)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # 不能将序列乘以非int 解决方式是将其转换为array数组就可以了
            weights = weights + alpha * error * array(dataMatrix[randIndex])
            del(dataInx[randIndex])
    return weights

# 测试算法：用Logistic回归进行分类
def classifyVector(inx,weights):
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 采用修改过后的随机梯度上升函数,进行权重计算
    trainingWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" %(errorRate))
    return errorRate

def mutiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the averge error rate is: %f" %(numTests,errorSum/float(numTests)))


# 画出数据集合Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s = 30,c = 'green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
