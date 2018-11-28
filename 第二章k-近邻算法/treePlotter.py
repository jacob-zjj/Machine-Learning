import matplotlib.pyplot as plt
# 定义文本框和箭头格式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
arrow_args = dict(arrowstyle = "<-")

# 控制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',xytext = centerPt,textcoords = 'axes fraction',va = 'center',ha = 'center',bbox = nodeType,arrowprops = arrow_args)

# 在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    # 在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

# 创建决策树
def plotTree(myTree,parentPt,nodeTxt):
    # 计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    # 标记节点属性值
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    # 减少y偏移
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),cntrPt,leafNode)
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD


# 生成决策树图
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    # createPlot.ax1 = plt.subplot(111,frameon = False)
    # plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
    # plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
    axprops = dict(xticks = [],yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


# 获取叶节点的数目和树的层数
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def getNumLeafs(myTree):
    numLeafs = 0
    # print(myTree.keys())
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # print(secondDict.keys())
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 提前创建好树
def retrieveTree(i):
    listofTrees = [{'no surfacing': {0: 'no',1: {'flippers': {0: 'no',1: 'yes'}}}},{'no surfacing': {0: 'no',1: {'flippers': {0: {'head': {0: 'no',1: 'yes'}},1: 'no'}}}}]
    return listofTrees[i]
