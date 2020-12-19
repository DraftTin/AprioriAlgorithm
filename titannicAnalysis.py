import pandas as pd
from apriori import *

def getTransactionList(data):
    transactionList = list()
    for i in range(len(data)):
        transaction = set(data.iloc[i])
        transactionList.append(transaction)
    return transactionList

def getItemSetFromTransactionList(transactionList):
    """提取事务中的事务项, 返回事务项列表"""
    itemSet = set()
    for transaction in transactionList:
        for item in transaction:
            itemSet.add(frozenset([item]))
    return itemSet

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
    itemC = {'3rd', 'Male', 'Adult', 'Yes'}
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
    print(support / len(transactionList))

def homework_5():
    data = pd.read_csv('Titanic.csv')
    item = ({'Crew', 'No', 'Adult'}, {'Male'})
    transactionList = getTransactionList(data)
    support = getSupport(item[0].union(item[1]), transactionList)
    confidence = support / getSupport(item[0], transactionList)
    lift = confidence / (getSupport(item[1], transactionList) / len(transactionList))
    print('support: ', support / len(transactionList))
    print('confidence: ', confidence)
    print('lift: ', lift)

def homework_6():
    """检查数据集的规律"""
    data = pd.read_csv('Titanic.csv')
    itemFileName = 'titannic_item.txt'
    ruleFileName = 'titannic_rule.txt'
    minSupport = 0.02
    minConfidence = 0.7
    # 生成事务列表, 生成项集合
    transactionList = getTransactionList(data)
    itemSet = getItemSetFromTransactionList(transactionList)
    rItems, rRules = processApriori(itemSet, transactionList, minSupport, minConfidence)
    itemFile = open(itemFileName, 'w+')
    ruleFile = open(ruleFileName, 'w+')
    for item in rItems:
        itemFile.write("%s : %f\n" % (str(item[0]), item[1]))
    for rule in rRules:
        ruleFile.write("%s -----> %s : %f\n" % (str(rule[0][0]), str(rule[0][1]), rule[1]))

    itemFile.close()
    ruleFile.close()

if __name__ == '__main__':
    homework_6()
