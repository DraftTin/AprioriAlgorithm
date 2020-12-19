# coding=gbk

import jieba
import jieba.posseg as pseg
import pandas as pd
from collections import defaultdict
from itertools import chain, combinations

removed_words = {
    '�õ�', '����', '��Ϊ', 'ģ��', '����', '����', '����', '���',
    '����', '����', '�ҳ�', '�ṩ', '����', '����', '����',
    '�����ھ�', '���', '����', '����'
}

flagSet = {'n', 'v'}

def allSubsets(arr):
    """������������п��ܵ��Ӽ�"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def canAdd(word, flag):
    # removed_flag = {'x', 'u', 'm', 'ud', 'uj', 'ul', 'uv', 'uz', 'y', 'g', 'c', '     f', 'p', 'd', 'r'}
    return flag in flagSet and len(word) > 1 and word not in removed_words

def joinSet(itemSet, length):
    """��ȡ���Ϊlength���"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

# generateFrequentItemSet: ��������ɸ�Ƶ�
def generateFrequentItemSet(itemSet, transactionList, freqSet, minSupport):
    """"��itemSet�м�����µ�Ƶ���������"""
    supportSet = defaultdict(int)
    newItemSet = set()
    for item in itemSet:
        # ����support
        for transaction in transactionList:
            # ���item�������г�����support + 1
            if(item.issubset(transaction)):
                supportSet[item] += 1
                freqSet[item] += 1
            # ����֧�ֶ�
            support = supportSet[item] / len(transactionList)
            if support >= minSupport:
                newItemSet.add(item)
    return newItemSet

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
    return itemSet, transactionList


# processApriori: apriori�㷨����Ƶ����͹�������
# - ��ȡ���е�����, ��ȡ����������б�
# - ���ݵ�ǰ���������Ӧ��Ƶ���, ����
# - ���� k + 1 Ԫ�, ����, ѭ��ִ��ֱ�����ɵ�Ƶ���Ϊ��
# - ��������Ƶ�����������
# - ��ÿһ���ֳ������Ӽ�, �������Ŷ�, ���������Ŷȵķָʽ��¼, ����
# - ��������Ƶ����, ���б���Ĺ����������Ӧ��֧�ֶȺ����Ŷ�
def processApriori(itemSet, transactionList, minSupport=0.05, minConfidence=0.5):
    freqSet = defaultdict(int)      # ���ڼ�¼�������Ƶ��, ���������
    allFreqItemSet = dict()         # ���ڼ�¼���е�Ƶ���: Ƶ��һ�, Ƶ�����

    frequentOneSet = generateFrequentItemSet(itemSet, transactionList, freqSet, minSupport)
    currentLSet = frequentOneSet
    k = 2
    while currentLSet != set():
        allFreqItemSet[k - 1] = currentLSet
        # ��k��Ƶ����
        currentLSet = joinSet(currentLSet, k)
        currentCSet = generateFrequentItemSet(currentLSet, transactionList, freqSet, minSupport)
        currentLSet = currentCSet
        k += 1

    def getSupport(item):
        """����һ���֧�ֶ�"""
        return float(freqSet[item]) / len(transactionList)

    # ��ȡ����Ƶ������Ӧ��֧�ֶ�
    rItems = []
    for key, value in allFreqItemSet.items():
        rItems.extend([tuple(item), getSupport(item)] for item in value)
    # ��ȡ����Ƶ��������Ӧ�����Ŷ�
    rRules = []
    for key, value in list(allFreqItemSet.items())[1:]:
        # ����ÿ��Ƶ����������Ӽ�
        for item in value:
            subsetList = map(frozenset, [x for x in allSubsets(item)])
            for subset in subsetList:
                # ��
                remain = item.difference(subset)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(subset)
                    if confidence >= minConfidence:
                        rRules.append(((tuple(subset), tuple(remain)), confidence))
    return rItems, rRules

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


