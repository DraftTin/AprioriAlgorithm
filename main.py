# coding=gbk

import jieba.posseg as pseg
import pandas as pd
from apriori import *


removed_words = {
    '�õ�', '����', '��Ϊ', 'ģ��', '����', '����', '����', '���',
    '����', '����', '�ҳ�', '�ṩ', '����', '����', '����',
    '�����ھ�', '���', '����', '����'
}

flagSet = {'n', 'v'}

def canAdd(word, flag):
    # removed_flag = {'x', 'u', 'm', 'ud', 'uj', 'ul', 'uv', 'uz', 'y', 'g', 'c', '     f', 'p', 'd', 'r'}
    return flag in flagSet and len(word) > 1 and word not in removed_words

# getWordsList: �Զ�ȡ���ı����зִ�, ����ÿ�εķִʽ��
def getWordsList(data):
    wordsList = list()
    for item in data:
        if type(item) != str:
            continue
        words = pseg.cut(item)
        wordsList.append(words)
    return wordsList

def getItemSetTransactionListFromWordsList(wordsList):
    # ����������������б�
    transactionList = list()
    itemSet = set()
    for words in wordsList:
        # �Ƴ����õĴʻ�, �������, ��, ��
        transaction = set([word for word, flag in words if canAdd(word, flag)])
        transactionList.append(transaction)

        # ����һ�
        for item in transaction:
            itemSet.add(frozenset([item]))
    print(itemSet)
    return itemSet, transactionList

# �������ݲ������ļ�
def analyzeData(data, minSupport, minConfidence, plagiarismThresh,itemFileName, ruleFileName, plagiarismFileName):
    # ��ȡ����������зִ�֮��������б�, itemSet�е�����frozenset��ʽ, ��Ϊ֮��Ҫ���������
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

# findPlagiarisms: ���ҳ�Ϯ������, ���ر���Ϊ�ǳ�Ϯ��Ԫ��Ե��б�
def findPlagiarisms(data, rItems, plagiarismThresh):
    # ��ȡ���ɸ�Ƶ��
    highFrequencyItems = findItemForPlagiarisms(rItems)
    plagiarisms = set()
    for item in highFrequencyItems:
        # ʹ�ø�Ƶ����ҳ�Ϯ����
        tmp = findPlagiarismsFromItem(data, item, plagiarismThresh)
        plagiarisms = plagiarisms.union(tmp)
    return plagiarisms

# findItemForPlagiarisms: �������ڼ�鳭Ϯ�ĸ�Ƶ�ִ���, ����ѡ�����ʶ����
def findItemForPlagiarisms(rItems):
    items = set()
    maxLen = 0
    for item in rItems:
        maxLen = max(maxLen, len(item[0]))
    for item in rItems:
        if len(item[0]) == maxLen:
            items.add(item[0])
    return items

# findPlagiarismsFromItem: ���ݸ�Ƶ���ҵ���Ϯ����
# - �ҵ��������а�����Ƶ�������дʵĴ�, ����
# - ���������Щ���ݽ���ģ��ƥ��, ���ƥ��ȳ�����ֵ����Ϊ�ǳ�Ϯ
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
                # ���� + 2 ��Ϊ��һ��������
                plagiarisms.add((matchedData[i][1] + 2, matchedData[j][1] + 2))
    return plagiarisms



if __name__ == '__main__':
    """����ͬѧ�ύ����ҵ"""
    data = pd.read_excel('data.xls')
    dataA = data[data.columns[0]]
    dataB = data[data.columns[1]]
    analyzeData(dataA, 0.1, 0.6, 0.98, 'itemA.txt', 'ruleA.txt', 'plagiarismsA.txt')
    analyzeData(dataB, 0.1, 0.6, 0.8, 'itemB.txt', 'ruleB.txt', 'plagiarismsB.txt')


