def creatDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

# 计算给定数据的熵值，熵被定义为信息的期望值，为了计算熵我们需要计算所有类别所有可能值包含的信息期望值
from math import log
# math是进行的数字运算，也就是说得到的值是float类型的值
# cmath是进行的复数运算，得到值是复数类型的
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 依次对dataSet中[[1,1,'yes'],
    #                [1,0,'no'],
    #                [0,1,'no'],
    #                [0,1,'no']]
    # 对内部结果值进行统计
    for featVec in  dataSet:
        currentLabel = featVec[-1]
        # 下面这两句话的设计方式很巧妙，可以简单的对每部数据特征进行统计
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # H = -对p(xi)log2(p(xi))进行求和
    # 计算所有类别所有可能值包含的信息期望值
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 按照给定特征划分数据集
# 待划分的数据集    划分数据集的特征   需要返回的特征的值
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            # 存储出去指定axis下标所对应的数据
            retDataSet.append(reducedFeatVec)
    # 返回list类型的数据
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 记录除开最后一列的判断值的总的特征总数
    numFeatures = len(dataSet[0]) - 1
    # 计算基本熵值
    baseEntropy = calcShannonEnt(dataSet)
    # 首先将最佳熵值记为0.0
    bestInfoGain = 0.0
    # 以此作为最佳划分的最佳特征下标
    bestFeature = -1
    for i in range(numFeatures):
        # 分别将每个集合中的对应i位置的特征放入featList中
        featList = [example[i] for example in dataSet]
        # 去掉重复
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 这部分和KNN中的classfy0()方法相同都起到了类似的作用
# 返回出现次数最多的分类名称
import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

# 创建树的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # print(classList)
    """
    ['yes', 'yes', 'no', 'no', 'no']
    1
    ['no', 'no']
    2
    ['yes', 'yes', 'no']
    2.1
    ['no']
    ['yes', 'yes']
    """
    # 类别完全相同则停止继续划分
    # 判断决策树结束的条件1：
    # 也就是说对类别进行统计看起数量是否等于集合的长度
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 判断决策时结束的条件2：
    # 遍历完所有特征时返回出现次数最多的特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

# 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    # 按照二进制位对数据进行写入
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    # 按照二进制位对数据进行读取
    fr = open(filename,'rb')
    return pickle.load(fr)





