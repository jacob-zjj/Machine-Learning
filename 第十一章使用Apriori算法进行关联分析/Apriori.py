#Apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(t) for t in C1]
def scanD(D,CK,minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList,supportData

# Apriori算法
def aprioriGen(LK,K):
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            # 前K-2个项相同时，将两个集合合并
            L1 = list(LK[i])[:K-2]
            L2 = list(LK[j])[:K-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(LK[i]|LK[j])
    return retList
def apriori(dataSet,minSupport = 0.5):
    D = list(map(set, dataSet))
    C1 = createC1(dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    K = 2
    while (len(L[K-2]) > 0):
        CK = aprioriGen(L[K-2],K)
        LK,supK = scanD(D,CK,minSupport)
        supportData.update(supK)
        L.append(LK)
        K += 1
    return L,supportData

# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList,minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
# 对规则进行评估
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
# 生成候选规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

