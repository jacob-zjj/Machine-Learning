from numpy import *
import operator
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# inx为输入的准备预测的值； dataSet为数据集； labels为数据集对应的分类；k为选择最近邻居的数目
# labels中的元素个数和dataSet中矩阵行数相同
def classfy0(inx,dataSet,labels,k):
    # 输出dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 输出dataSet的列数
    # dataSetSize1 = dataSet.shape[1]

    # 计算距离的第一步，点与点的距离
    # d = [(xA0-XB0)**2 + (xA1-xB1)**2]**0.5
    diffMat = tile(inx,(dataSetSize,1)) - dataSet
    # title 表示重复某个数组
    # [ [-1. - 1.1]
    #   [-1. - 1.]
    #   [0.   0.]
    #   [0. - 0.1]]
    # 得到的diffMat表示将输入的预测[0,0]变成和dataSet维度相同的矩阵与dataSet相减得到的差值

    # 下一步是要对其进行取平方
    sqDiffMat = diffMat ** 2
    # [[1.   1.21]
    #  [1.   1.]
    #  [0.   0.]
    #  [0.   0.01]]

    sqDistances = sqDiffMat.sum(axis = 1)
    # axis表示对每一行进行求和
    # [2.21 2.   0.   0.01]

    distances = sqDistances ** 0.5
    # 对平方和进行开方

    sortedDisIndicies = distances.argsort()
    # 将距离从小到大排序，得到的是每个距离的下标
    # 表示该点距离给的dataSet中的几个点距离依次排序
    # [2 3 1 0]

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        # B B A
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 字典中的get方法可以有两个参数，第一参数是key,第二个是default
        # 当get找不到key时，就会在字典中新建这个key值，那么这个新建的Key值对应的value的值就是默认的值
        # {'B': 2, 'A': 1}

    # print(classCount.items())
    # dict_items([('B', 2), ('A', 1)])

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # key=operator.itemgetter(1) 表示按照迭代器中每个元素对应1下标中的值进行排序，reverse = True表示逆序
    # [('B', 2), ('A', 1)]

    return sortedClassCount[0][0]

# 将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    # 把文本读成不同的行
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 设置一个numberOfLines行，3列的矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 对文本中每行数据进行处理，将每行数据读入矩阵中
        line = line.strip()
        listFormLine = line.split('\t')
        # 下标中的index,：表示在index行中的所有列信息
        returnMat[index,:] = listFormLine[0:3]
        # 将文本中的最后一行的数据代表属性标识的数据写入单独的1行n列的矩阵中
        classLabelVector.append(int(listFormLine[-1]))
        index = index + 1
    return returnMat,classLabelVector

# 将数值归一化，该函数自动将数字特征值转化为0到1的区间
def autoNorm(dataSet):
    # 由于dataSet是一个矩阵，因此min和max都是相当于求矩阵中每列的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # 生成和datasSet相同形式的零矩阵
    normDataSet = zeros(shape(dataSet))
    # m为dataSet的行数
    m = dataSet.shape[0]
    # tile(minVals,(m,1))是将minVal变成m行minVals列的矩阵
    normDataSet = dataSet - tile(minVals,(m,1))
    # 两个矩阵作商，得到归一化的矩阵
    normDataSet = normDataSet / tile(ranges,(m,1))
    return  normDataSet,ranges,minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    # 得到归一化矩阵的行数
    m = normMat.shape[0]
    # 取百分之10作为测试数据
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 测试过程中将前10%行的数据作为测试数据进行逐一处理，每次要将i行的数据和整体的数据进行Classfy0中进行预测
        classifierResult = classfy0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # 打印预测值和真实值，看出这两者之间的差距
        print("The classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i]))
        # 如果不相等将错误值加1
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f " %(errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("Percentage of time spent playing video games? \n"))
    ffMiles = float(input("Frequence flier miles earned per year? \n"))
    iceCream = float(input("liters of ice cream consumed per year? \n"))
    datingDataMat,datingLabels = file2matrix('F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    # 预测该男士的结果为 1  2  3
    classfierResult = classfy0((inArr - minVals)/ranges,norMat,datingLabels,3)
    # 结果 - 1 对应 resultLIs
    print("You will probably like this person: ",resultList[classfierResult - 1])

# 下面部分为测试手写数字的代码
# 将二进制图像转换为向量：该函数创建一个1x1024的Numpy数组，然后打开给定的文件，循环读取
def img2vector(filename):
    # 原始二进制图像编码为32 x 32 = 1024 的数组，因此用1行1024列进行重新存储作为矩阵输出
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    return returnVect


# 测试算法：使用k-近邻算法识别手写数字
# 列出给定目录的文件名
from os import listdir
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\trainingDigits')
    # ['0_0.txt', '0_1.txt', '0_10.txt', '0_100.txt', '0_101.txt', '0_102.txt', '0_103.txt', '0_104.txt', '0_105.txt',
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\trainingDigits\%s' %(fileNameStr))
    testFileList = listdir(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\testDigits\%s' %(fileNameStr))
        classiResult = classfy0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d ,the real answer is: %d." %(classiResult,classNumStr))
        if (classiResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" %(errorCount))
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))
