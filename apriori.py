from collections import defaultdict
from itertools import chain, combinations

def allSubsets(arr):
    """返回数组的所有可能的子集"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def joinSet(itemSet, length):
    """获取项长度为length的项集"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

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