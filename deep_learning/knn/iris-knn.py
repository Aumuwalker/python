import pandas as pd
from csv import reader
from random import random
from math import sqrt
import operator


'''
k-近邻算法
'''
class IrisKnn:

    def __init__(self):
        pass

    def write_to_csv(self, url, filename, csv_header):
        '''
        将文件保存到本地， 格式为csv
        :param url: 文件所在url
        :param filename: 需要写到的目标文件文件路径
        :param csv_header: 文件头
        :return: 无返回值
        '''
        df = pd.read_csv(url)
        df.to_csv(filename, index=False, encoding='UTF-8')

    def readdata(self, filename, split, trainSet=[], testSet=[]):
        '''
        读取本地文件
        :param filename: 需要读取的文件
        :param split: 将数据划分的比例
        :param trainSet: 训练数据集合
        :param testSet: 测试数据集合
        :return: 无返回值
        '''
        with open(filename, 'r') as f:
            lines = reader(f)
            dataset = list(lines)  # 将数据转换为列表
            row = len(dataset)  # 数据行数
            col = len(dataset[0])-1  # 数据列数
            for x in range(row):
                for y in range(col):
                    dataset[x][y] = float(dataset[x][y])
                if random() < split:
                    trainSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])

    def EuclidDist(self, testData, trainData, length):
        '''
        计算相似性，采用欧几里得距离
        :param testData: 单个测试数据
        :param trainData: 单个训练数据
        :param length: 数据特征值个数
        :return: 返回两个数据之间的距离，类型为浮点数
        '''
        distance = 0.0
        for x in range(length):
            distance += pow(testData[x]-trainData[x], 2)
        return sqrt(distance)

    def getNeighbors(self, trainSet, testData, k):
        '''
        针对单个测试数据在所有训练数据中找到最近的k各邻居
        :param trainSet: 训练数据集合
        :param testData: 测试数据
        :param k: 需要寻找的邻居个数
        :return: 类型为列表，列表中元素为该数据的邻居
        '''
        distances = []
        for x in range(len(trainSet)):
            distance = self.EuclidDist(testData, trainSet[x], len(trainSet[0])-1)
            distances.append((trainSet[x], distance))
        distances.sort(key=operator.itemgetter(1))  # 按照distances里下标为一的元素排序
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getClass(self, neighbors):
        '''
        对数据进行分类
        :param neighbors: 数据的最近邻居
        :return: 返回数据的类别
        '''
        classRet = {}  # 数据类别作为键，邻居个数作为值
        classSort = []
        for x in range(len(neighbors)):
            if neighbors[x][-1] in classRet:
                classRet[neighbors[x][-1]] += 1
            else:
                classRet[neighbors[x][-1]] = 1
            classSort = sorted(classRet.items(), key=operator.itemgetter(1), reverse=True)
        return classSort[0][0]

    def getPrediction(self, trainSet, testSet, k):
        '''
        通过训练数据集合对测试数据集合进行预测
        :param trainSet: 训练数据集合
        :param testSet: 测试数据集合
        :param k: 邻居个数
        :return: 预测数据集合
        '''
        prediction = []
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(trainSet, testSet[x], k)
            prediction.append(self.getClass(neighbors))
        return prediction

    def getAccuracy(self, testSet, prediction):
        '''
        得到预测的准确度
        :param testSet: 训练数据集合
        :param prediction: 预测数据集合
        :return: 正确率
        '''
        if len(testSet) != len(prediction):
            print('参数长度不相等')
            return
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == prediction[x]:
                correct += 1
        return (correct/float(len(testSet)))*100

if __name__ == '__main__':
    trainSet = []  # 训练数据集
    testSet = []  # 测试数据集
    knn = IrisKnn()
    knn.readdata('test.csv', 0.7, trainSet, testSet)
    prediction = knn.getPrediction(trainSet, testSet, 3)
    correct_rate = knn.getAccuracy(testSet, prediction)
    print(correct_rate)