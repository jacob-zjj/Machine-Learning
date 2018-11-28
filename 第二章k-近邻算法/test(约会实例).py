import KNN
from numpy import *
from numpy import array
# 得到转化后的三种特征数据 和 一种标识数据
datingDataMat,datingLabels = KNN.file2matrix('F:\Deeplearning\机器学习实战10月25日\MLAdata\Ch02\datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)
# print(array(datingLabels))
# 数据可视化
# 第一步
'''
# 通过比较可视化第二列玩游戏和第三列买冰淇淋的分类
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax  = fig.add_subplot(111)
# 将得到的数据的第二列和第三列对应的值通过图表的形式可视化出来
# 将dataingLables转换成numpy.array数组。在可视化的时候图表出现不同的颜色有便于分类
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.xlabel('Play games')
plt.ylabel('IceCream')
plt.show()
'''
# 第二步
'''
通过比较将得出将第2列和第3列作为分类的条件没有把第1列和第二列作为分类条件这样将数据分类得更加准确

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax  = fig.add_subplot(111)
# 将得到的数据的第二列和第三列对应的值通过图表的形式可视化出来
# 将dataingLables转换成numpy.array数组。在可视化的时候图表出现不同的颜色有便于分类
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.xlabel('Mile of Flight')
plt.ylabel('IceCream')
plt.show()
'''
# 第三步
# 得到转换后的矩阵和范围以及每列最小值
norMat, ranges, minVals = KNN.autoNorm(datingDataMat)
# 转换最后我们就可以将以此作为评价基准
# print(norMat)
# print(ranges)
# print(minVals)

# 第四步
# 得出预测值的误差大小通过与标准值的比较
# KNN.datingClassTest()
# 得出模型的误差为5%，该模型可行

# 第五步
# 完整的预测模型使用
KNN.classifyPerson()

