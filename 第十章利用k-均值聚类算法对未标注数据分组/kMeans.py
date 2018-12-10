from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat
def disEclud(vecA,vecB):
    import math
    # sqrt是通过math导入的，power是通过numpy导入的
    return math.sqrt(sum(power(vecA - vecB,2)))
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

# K-均值聚类算法
def Kmeans(dataSet,k,distMeans = disEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,:] != minIndex:
                clusterChanged = True
            clusterAssment[i,0] != minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis = 0)
    return centroids,clusterAssment

# 二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas = disEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss = Kmeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print("sseSplit, and not Split: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print("the bestCentToSplit is: ",bestCentToSplit)
        print("the len of bestClustAss is: ",len(bestClustAss))
        # 此处应该先修改为list,然后进行赋值和添加，先前的bestNewCents为materix矩阵
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment
