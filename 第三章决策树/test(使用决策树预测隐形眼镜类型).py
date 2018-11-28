import trees
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# print(lenses)
lensesLabels = ['age','prescript','astigmatic','tearRate']
# 创建决策树
lensesTree = trees.createTree(lenses,lensesLabels)
import treePlotter
# 创建决策树图
treePlotter.createPlot(lensesTree)
