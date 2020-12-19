# coding=gbk

import jieba.posseg as pseg
import pandas as pd
from apriori import *


removed_words = {
    '用到', '满足', '行为', '模型', '可能', '建立', '减少', '提高',
    '进行', '产生', '找出', '提供', '产生', '大量', '问题',
    '数据挖掘', '辨别', '存在', '发现'
}

flagSet = {'n', 'v'}

def canAdd(word, flag):
    # removed_flag = {'x', 'u', 'm', 'ud', 'uj', 'ul', 'uv', 'uz', 'y', 'g', 'c', '     f', 'p', 'd', 'r'}
    return flag in flagSet and len(word) > 1 and word not in removed_words

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
    print(itemSet)
    return itemSet, transactionList

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



if __name__ == '__main__':
    """分析同学提交的作业"""
    data = pd.read_excel('data.xls')
    dataA = data[data.columns[0]]
    dataB = data[data.columns[1]]
    analyzeData(dataA, 0.1, 0.6, 0.98, 'itemA.txt', 'ruleA.txt', 'plagiarismsA.txt')
    analyzeData(dataB, 0.1, 0.6, 0.8, 'itemB.txt', 'ruleB.txt', 'plagiarismsB.txt')


