import trees
myDat,labels = trees.creatDataSet()
# 得到myDat数据的熵值
# print(trees.calcShannonEnt(myDat))
# print(labels)

"""
print(trees.splitDataSet(myDat,0,1))
找到myDat中每个list中下标为0对应的值为1的表单，将其除了0位置以外的其他位置重新存储再返回给用户
"""

"""
测试那个数据为最佳数据
# print(trees.chooseBestFeatureToSplit(myDat))
"""


"""
创建决策树数据进行决策分类
print(trees.createTree(myDat,labels))
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
"""

import treePlotter
"""
绘制决策树:
treePlotter.createPlot()
"""

"""
获取树的叶子数量和深度:
myTree = treePlotter.retrieveTree(0)
print(treePlotter.getNumLeafs(myTree))
print(treePlotter.getTreeDepth(myTree))
"""

"""
生成决策图:
myTree = treePlotter.retrieveTree(0)
myTree['no surfacing'][3] = 'maybe'
treePlotter.createPlot(myTree)
"""

"""
测试算法:
myTree = treePlotter.retrieveTree(0)
print(trees.classify(myTree,labels,[1,0]))
print(trees.classify(myTree,labels,[1,1]))
"""

"""
使用pickle模块存储决策树
myTree = treePlotter.retrieveTree(0)
trees.storeTree(myTree,'classifierStorage.txt')
print(trees.grabTree('classifierStorage.txt'))
"""


