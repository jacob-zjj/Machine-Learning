"""
测试转换函数
import bayes
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
"""

"""
试验得到两种类型的概率，以及辱骂文档的概率
import bayes
list0Posts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(list0Posts)
trainMat = []
for postinDoc in list0Posts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
p0V,p1V,pAb = bayes.trainNBO(trainMat,listClasses)
print(pAb)
print(p0V)
print(p1V)
"""

"""
测试贝叶斯算法
"""
import bayes
bayes.testingNB()

