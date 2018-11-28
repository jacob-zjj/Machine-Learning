# 词表到向量的转换函数
# 创建数据集
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['mybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['qiut','buying','worthless','dog','food','stupid']
    ]
    classVec = [0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec

# 创建非重复词表
def createVocabList(dataSet):
    vacabSet = set([])
    # vacabSet就是一个没有重复单词的集合。因此在进行转换时需要将ducument转换成set模式以便进行
    for ducument in dataSet:
        vacabSet = vacabSet | set(ducument)
    return vacabSet

# 对单词进行统计同时对词表统计数组中的单词数量进行更新
def setOfWords2Vec(vocabList,inputSet):
    vocabList = list(vocabList)
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec

# 准备数据：文档词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 朴素贝叶斯分类训练函数
from numpy import *
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 辱骂性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 向量相加
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算出两种类型的概率
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
from math import log
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # 这里的相乘指的是两个向量相称的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘
    # 然后再将第二个元素相乘，依次类推
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

from numpy import array
def testingNB():
    list0posts,listClasses = loadDataSet()
    myVocabList = createVocabList(list0posts)
    trainMat = []
    for postinDoc in list0posts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNBO(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classify as: ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classify as: ",classifyNB(thisDoc,p0V,p1V,pAb))

# 测试算法：使用朴素贝叶斯进行交叉验证
# 文件解析及完整的垃圾邮件测试函数
    def textParse(bigString):
        import re
        listOfTokens = re.split(r'\W*',bigString)
        return [tok.lower() for tok in  listOfTokens if len(tok) > 2]

    # 样本的测试
    def spamTest():
        docList = []
        classList = []
        fullText = []
        for i in range(1,26):
            # 垃圾邮件标记为1并且存储在1对应的位置
            wordList = textParse(open(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch04\email\spam\%d.txt' %i).read())
            docList.append(wordList)
            fullText.append(wordList)
            classList.append(1)
            # 有用的邮箱标记为0并且存储在0对应的位置
            wordList = textParse(open(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch04\email\ham\%d.txt' %i).read())
            docList.append(wordList)
            fullText.append(wordList)
            classList.append(0)
            # classList = [1,0,1,0,1,0]
        # 创建非重复词表
        vocabList = createVocabList(docList)
        # 创建测试集合
        trainingSet = range(50)
        testSet = []
        import random
        for i in range(10):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in  trainingSet:
            trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNBO(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = setOfWords2Vec(vocabList,docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
        print("the error rate is : ",float(errorCount)/len(testSet))

# RSS源分类及高频词去除函数
    def calMostFreq(vocabList,fullText):
        import operator
        freqDict = {}
        for token in vocabList:
            freqDict[token] = fullText.count(token)
        sortedFreq = sorted(freqDict.items(),key = operator.itemgetter(1),reverse = True)
        return sortedFreq[:30]

    def localWords(feed1,feed0):
        import feedparser
        docList = []
        classList = []
        fullText = []
        minLen = min(len(feed1['entries']),len(feed0['entries']))
        for i in range(minLen):
            wordList = textParse(feed1['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = textParse(feed0['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = createVocabList(docList)
        top3Words = calMostFreq(vocabList,fullText)
        for pariw in top3Words:
            if pariw[0] in vocabList:
                vocabList.remove(pariw[0])
        trainingSet = list(range(2*minLen))
        testSet = []
        import random
        for i in range(20):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNBO(array(trainMat),array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
        print("the error rate is: ",float(errorCount)/len(testSet))
        return vocabList,p0V,p1V