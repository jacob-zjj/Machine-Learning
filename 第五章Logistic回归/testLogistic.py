import logRegres
"""
# 测试梯度上升优化算法
dataArr,labelMat = logRegres.loadDataSet()
print(logRegres.gradAscent(dataArr,labelMat))
"""

"""
# 分析数据：画出决策边界
# from numpy import *
dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr,labelMat)
# weights.getA()的方法是将numpy数组转换为矩阵
logRegres.plotBestFit(weights.getA())
"""
# from numpy import *
dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr,labelMat)
# weights.getA()的方法是将numpy数组转换为矩阵
logRegres.plotBestFit(weights.getA())

"""
# 测试随机梯度上升算法
dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent0(dataArr,labelMat)
logRegres.plotBestFit(weights)
"""

"""
# 改进的随机梯度上升算法
dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent1(dataArr,labelMat)
logRegres.plotBestFit(weights)
"""


"""
# 使用测试函数对数据进行测试返回错误率
"""
# logRegres.mutiTest()



