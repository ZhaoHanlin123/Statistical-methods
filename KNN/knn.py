"""
This is the implementation of Knn(KdTree),
which is accessible in https://github.com/FlameCharmander/MachineLearning,
accomplished by FlameCharmander,
and my csdn blog is https://blog.csdn.net/tudaodiaozhale,
contact me via 13030880@qq.com.
"""
import numpy as np

# 创建结点类
class Node:
    def __init__(self, data, lchild = None, rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


# 创建KdTree类
class KdTree:
    def __init__(self):
        self.kdTree = None

    # 创建kd树，返回跟结点
    def create(self, dataSet, depth):
        if(len(dataSet)) > 0:  # 如果数据集非空则生成结点
            m, n = np.shape(dataSet)  # 求出数据集的行和列，行为m, 列为n
            midindex = int(m / 2)  # 求取中间数的索引位置
            axis = depth % n  # 判断以哪一个轴来划分数据集
            sortedDataSet = self.sort(dataSet, axis)  # 对axis进行排序
            node = Node(sortedDataSet[midindex])  # 将中位数数据设置为节点
            leftDataSet = sortedDataSet[: midindex]  # 分别对中位数左右数据创建数据集
            rightDataSet = sortedDataSet[midindex+1 :]
            print(leftDataSet)
            print(rightDataSet)
            node.lchild = self.create(leftDataSet, depth+1)  # 递归创建树
            node.rchild = self.create(rightDataSet, depth+1)
            return node
        else:
            return None

    # 采用冒泡排序，利用aixs作为轴进行划分
    def sort(self, dataSet, axis):
        sortDataSet = dataSet[:]  # 由于不能破坏原样本，此处建立一个副本
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(0, m - i -1):
                if (sortDataSet[j][axis] > sortDataSet[j+1][axis]):
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j+1]
                    sortDataSet[j+1] = temp
        print(sortDataSet)
        return sortDataSet

    def preOrder(self, node):
        if node != None:
            print("tttt->%s" % node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    # 搜索KdTree
    def search(self, tree, x):
        self.nearestPoint = None  # 保存最近的点
        self.nearestValue = 0  # 保存最近的值

        def travel(node, depth=0):  # 递归搜索
            if node is not None:  # 递归终止条件
                n = len(x)  # 特征数
                axis = depth % n  # 计算轴
                if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找,反之，往右结点找
                    travel(node.lchild, depth+1)
                else:
                    travel(node.rchild, depth+1)
                # 以下是递归完毕后，往父结点方向回溯
                distNodeAndX = self.dist(x, node.data)  # 目标和结点的距离判断
                if (self.nearestPoint == None):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                print(node.data, depth, self.nearestValue, node.data[axis], x[axis])

                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth+1)  # x位于左结点 所以去右结点查找
                    else:
                        travel(node.lchild, depth+1)  # x位于右结点 所以去左结点查找

        travel(tree)
        return self.nearestPoint

    # 计算欧式距离
    def dist(self, x1, x2):
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5


if __name__ == "__main__":

    dataSet = [[2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [8, 1],
               [7, 2]]
    x = [5, 3]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)
    kdtree.preOrder(tree)
    print(kdtree.search(tree, x))