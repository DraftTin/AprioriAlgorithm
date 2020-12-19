# coding=gbk

import jieba
import jieba.posseg as pseg
import pandas as pd
from collections import defaultdict
from itertools import chain, combinations

removed_words = {
    '用到', '满足', '行为', '模型', '可能', '建立', '减少', '提高',
    '进行', '产生', '找出', '提供', '产生', '大量', '问题',
    '数据挖掘', '辨别', '存在', '发现'
}

flagSet = {'n', 'v'}

def allSubsets(arr):
    """返回数组的所有可能的子集"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def canAdd(word, flag):
    # removed_flag = {'x', 'u', 'm', 'ud', 'uj', 'ul', 'uv', 'uz', 'y', 'g', 'c', '     f', 'p', 'd', 'r'}
    return flag in flagSet and len(word) > 1 and word not in removed_words

def joinSet(itemSet, length):
    """获取项长度为length的项集"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

# generateFrequentItemSet: 根据项集生成高频项集
def generateFrequentItemSet(itemSet, transactionList, freqSet, minSupport):
    """"从itemSet中计算出新的频繁项集并返回"""
    supportSet = defaultdict(int)
    newItemSet = set()
    for item in itemSet:
        # 计算support
        for transaction in transactionList:
            # 如果item在事务中出现则support + 1
            if(item.issubset(transaction)):
                supportSet[item] += 1
                freqSet[item] += 1
            # 计算支持度
            support = supportSet[item] / len(transactionList)
            if support >= minSupport:
                newItemSet.add(item)
    return newItemSet

# getWordsList: 对读取的文本进行分词, 返回每段的分词结果
def getWordsList(data):
    wordsList = list()
    for item in data:
        if type(item) != str:
            continue
        words = pseg.cut(item)
        wordsList.append(words)
    return wordsList

def getItemSetTransactionListFromWordsList(wordsList):
    # 返回所有项和事务列表
    transactionList = list()
    itemSet = set()
    for words in wordsList:
        # 移除无用的词汇, 如标点符号, 的, 等
        transaction = set([word for word, flag in words if canAdd(word, flag)])
        transactionList.append(transaction)

        # 生成一项集
        for item in transaction:
            itemSet.add(frozenset([item]))
    return itemSet, transactionList


# processApriori: apriori算法计算频繁项集和关联规则
# - 获取所有单个项, 获取所有事务的列表
# - 根据当前的项集生成相应的频繁项集, 保存
# - 生成 k + 1 元项集, 保存, 循环执行直到生成的频繁项集为空
# - 遍历所有频繁项集的所有项
# - 将每一项拆分成两个子集, 计算置信度, 将满足置信度的分割方式记录, 保存
# - 返回所有频繁项, 所有保存的关联规则和相应的支持度和置信度
def processApriori(itemSet, transactionList, minSupport=0.05, minConfidence=0.5):
    freqSet = defaultdict(int)      # 用于记录所有项的频率, 包括组合项
    allFreqItemSet = dict()         # 用于记录所有的频繁项集: 频繁一项集, 频繁二项集

    frequentOneSet = generateFrequentItemSet(itemSet, transactionList, freqSet, minSupport)
    currentLSet = frequentOneSet
    k = 2
    while currentLSet != set():
        allFreqItemSet[k - 1] = currentLSet
        # 求k项频繁集
        currentLSet = joinSet(currentLSet, k)
        currentCSet = generateFrequentItemSet(currentLSet, transactionList, freqSet, minSupport)
        currentLSet = currentCSet
        k += 1

    def getSupport(item):
        """计算一项的支持度"""
        return float(freqSet[item]) / len(transactionList)

    # 获取所有频繁项及其对应的支持度
    rItems = []
    for key, value in allFreqItemSet.items():
        rItems.extend([tuple(item), getSupport(item)] for item in value)
    # 获取所有频繁项集和其对应的置信度
    rRules = []
    for key, value in list(allFreqItemSet.items())[1:]:
        # 生成每个频繁项集的所有子集
        for item in value:
            subsetList = map(frozenset, [x for x in allSubsets(item)])
            for subset in subsetList:
                # 求差集
                remain = item.difference(subset)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(subset)
                    if confidence >= minConfidence:
                        rRules.append(((tuple(subset), tuple(remain)), confidence))
    return rItems, rRules

# 分析数据并生成文件
def analyzeData(data, minSupport, minConfidence, plagiarismThresh,itemFileName, ruleFileName, plagiarismFileName):
    # 获取所有项和所有分词之后事务的列表, itemSet中的项是frozenset形式, 因为之后要处理组合项
    wordsList = getWordsList(data)
    itemSet, transactionList = getItemSetTransactionListFromWordsList(wordsList)
    rItems, rRules = processApriori(itemSet, transactionList, minSupport, minConfidence)
    itemFile = open(itemFileName, 'w+')
    ruleFile = open(ruleFileName, 'w+')
    plagiarismFile = open(plagiarismFileName, 'w+')
    for item in rItems:
        itemFile.write("%s : %f\n" % (str(item[0]), item[1]))
    for rule in rRules:
        ruleFile.write("%s -----> %s : %f\n" % (str(rule[0][0]), str(rule[0][1]), rule[1]))


    plagiarisms = findPlagiarisms(data, rItems, plagiarismThresh)
    for item in plagiarisms:
        plagiarismFile.write(str(item) + '\n')
    itemFile.close()
    ruleFile.close()
    plagiarismFile.close()

# findPlagiarisms: 查找抄袭现象函数, 返回被认为是抄袭的元组对的列表
def findPlagiarisms(data, rItems, plagiarismThresh):
    # 获取若干高频项
    highFrequencyItems = findItemForPlagiarisms(rItems)
    plagiarisms = set()
    for item in highFrequencyItems:
        # 使用高频项查找抄袭现象
        tmp = findPlagiarismsFromItem(data, item, plagiarismThresh)
        plagiarisms = plagiarisms.union(tmp)
    return plagiarisms

# findItemForPlagiarisms: 返回用于检查抄袭的高频分词组, 尽量选包含词多的项
def findItemForPlagiarisms(rItems):
    items = set()
    maxLen = 0
    for item in rItems:
        maxLen = max(maxLen, len(item[0]))
    for item in rItems:
        if len(item[0]) == maxLen:
            items.add(item[0])
    return items

# findPlagiarismsFromItem: 根据高频项找到抄袭现象
# - 找到包含所有包含高频项中所有词的答案, 保存
# - 将保存的这些内容进行模糊匹配, 如果匹配度超过阙值则认为是抄袭
def findPlagiarismsFromItem(data, targetWords, thresh=0.98):
    from fuzzywuzzy import fuzz

    matchedData = []
    k = -1
    for s in data:
        k += 1
        if type(s) != str:
            continue
        flag = True
        for word in targetWords:
            if word not in s:
                flag = False
                break
        if flag == True:
            matchedData.append((s, k))

    plagiarisms = set()
    for i in range(len(matchedData)):
        for j in range(len(matchedData)):
            if i == j or (j, i) in plagiarisms:
                continue
            similarityRate = fuzz.ratio(matchedData[i][0], matchedData[j][0])
            similarityRate = (float)(similarityRate) / 100
            if similarityRate >= thresh:
                # print("\n1. ", matchedData[i][0], matchedData[i][1], "\n2. ", matchedData[j][0], matchedData[j][1])
                # 这里 + 2 因为第一行是问题
                plagiarisms.add((matchedData[i][1] + 2, matchedData[j][1] + 2))
    return plagiarisms


def getTransactionList(data):
    transactionList = list()
    for i in range(len(data)):
        transaction = set(data.iloc[i])
        transactionList.append(transaction)
    return transactionList

def getSupport(item, transactionList):
    count = 0
    for transaction in transactionList:
        if item.issubset(transaction):
            count += 1
    return count

def homework_1():
    data = pd.read_csv('Titanic.csv')
    itemA = {'3rd', 'Male', 'Adult', 'No'}
    itemB = {'Crew', 'Male', 'Adult', 'Yes'}
    itemC = {'Crew', 'Male', 'Adult', 'No'}
    itemD = {'2nd', 'Male', 'Adult', 'No'}
    transactionList = getTransactionList(data)
    supportA = getSupport(itemA, transactionList)
    supportB = getSupport(itemB, transactionList)
    supportC = getSupport(itemC, transactionList)
    supportD = getSupport(itemD, transactionList)
    print("itemA: ", supportA)
    print("itemB: ", supportB)
    print("itemC: ", supportC)
    print("itemD: ", supportD)

def homework_2():
    data = pd.read_csv('Titanic.csv')
    itemA = {'1st', 'Female', 'Adult', 'Yes'}
    itemB = {'2nd', 'Female', 'Adult', 'Yes'}
    itemC = {'3rd', 'Male', 'Adult', 'No'}
    itemD = {'Crew', 'Male', 'Adult', 'Yes'}
    transactionList = getTransactionList(data)
    supportA = getSupport(itemA, transactionList)
    supportB = getSupport(itemB, transactionList)
    supportC = getSupport(itemC, transactionList)
    supportD = getSupport(itemD, transactionList)
    print("itemA: ", supportA)
    print("itemB: ", supportB)
    print("itemC: ", supportC)
    print("itemD: ", supportD)

def homework_3():
    data = pd.read_csv('Titanic.csv')
    itemA = ({'Crew', 'No'}, {'Male'})
    itemB = ({'3rd', 'No'}, {'Adult'})
    itemC = ({'2nd'}, {'Adult'})
    itemD = ({'3rd', 'Male', 'Adult'}, {'No'})
    transactionList = getTransactionList(data)
    confidenceA = getSupport(itemA[0].union(itemA[1]), transactionList) / getSupport(itemA[0], transactionList)
    confidenceB = getSupport(itemB[0].union(itemB[1]), transactionList) / getSupport(itemB[0], transactionList)
    confidenceC = getSupport(itemC[0].union(itemC[1]), transactionList) / getSupport(itemC[0], transactionList)
    confidenceD = getSupport(itemD[0].union(itemD[1]), transactionList) / getSupport(itemD[0], transactionList)
    liftA = confidenceA / (getSupport(itemA[1], transactionList) / len(transactionList))
    liftB = confidenceB / (getSupport(itemB[1], transactionList) / len(transactionList))
    liftC = confidenceC / (getSupport(itemC[1], transactionList) / len(transactionList))
    liftD = confidenceD / (getSupport(itemD[1], transactionList) / len(transactionList))
    print('liftA: ', liftA)
    print('liftB: ', liftB)
    print('liftC: ', liftC)
    print('liftD: ', liftD)

def homework_4():
    data = pd.read_csv('Titanic.csv')
    item = {'3rd', 'Male', 'Child', 'No'}
    transactionList = getTransactionList(data)
    support = getSupport(item, transactionList)
    print(100 * support / len(transactionList))

def homework_5():
    data = pd.read_csv('Titanic.csv')
    item = ({'Crew', 'No', 'Adult'}, {'Male'})
    transactionList = getTransactionList(data)
    support = getSupport(item[0].union(item[1]), transactionList)
    confidence = support / getSupport(item[0], transactionList)
    lift = confidence / (getSupport(item[1], transactionList) / len(transactionList))
    print('support: ', 100 * support / len(transactionList))
    print('confidence: ', 100 * confidence)
    print('lift: ', lift)

if __name__ == '__main__':
    data = pd.read_excel('data.xls')
    dataA = data[data.columns[0]]
    dataB = data[data.columns[1]]
    analyzeData(dataA, 0.1, 0.6, 0.98, 'itemA.txt', 'ruleA.txt', 'plagiarismsA.txt')
    analyzeData(dataB, 0.1, 0.6, 0.8, 'itemB.txt', 'ruleB.txt', 'plagiarismsB.txt')
    # homework_5()


