'''
一元线性回归预测
'''
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt


# 从csv文件中读取x，y的值放入列表中
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as f:
        csv_reader = reader(f)
        # 读取表中的第一行数据x和y
        head = next(csv_reader)
        # 文件指针下移到用于预测的数据
        for row in csv_reader:
            # 如果此行为空，则跳过
            if not row:
                continue
            dataset.append(row)
        return dataset


# 将读取到的数据转换为可计算的浮点数
def str_to_float(dataset):
    col_num = len(dataset[0])
    col = 0
    while col < col_num:
        for row in dataset:
            row[col] = float(row[col].strip())
        col += 1


# 将数据集分为训练集合和测试集合两部分
def train_test_split(dataset, percent):
    traindata = []
    train_size = percent*len(dataset)
    testdata = list(dataset)
    # 通过随机下标选择训练数据和测试数据
    while len(traindata) < train_size:
        index = randrange(len(testdata))
        traindata.append(testdata.pop(index))
    return traindata, testdata


# 计算样本均值
def mean(values):
    return sum(values) / float(len(values))


# 计算样本的方差
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])


# 计算样本的协方差
def covariance(x, x_mean, y, y_mean):
    count = 0.0
    for i in range(len(x)):
        count += (x[i] - x_mean)*(y[i] - y_mean)
    return count


# 计算回归系数
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[-1] for row in dataset]
    x_mean = mean(x)
    y_mean = mean(y)
    w1 = covariance(x, x_mean, y, y_mean)/variance(x, x_mean)
    w0 = y_mean - w1*x_mean
    return w1, w0


# 计算均方根误差
def caculate_rmse(actual, model):
    count = 0.0
    for i in range(len(actual)):
        count += (actual[i]-model[i])**2
    return sqrt(count/float(len(actual)))


# 构建线性回归
def linear_regression(traindata, testdata):
    predictions = []
    w1, w0 = coefficients(traindata)
    for row in testdata:
        x = row[0]
        y_predictions = w1*x + w0
        predictions.append(y_predictions)
    return predictions


# 评估算法
def evaluate_algorithm(dataset, algorithm, split_percent, *args):
    traindata, testdata = train_test_split(dataset, split_percent)
    model = algorithm(traindata, testdata, *args)
    actual = [row[-1] for row in testdata]
    rmse = caculate_rmse(actual, model)
    return rmse, traindata, testdata


# 画出训练数据和测试数据的散点图和建模后的预测数据函数图
def drawdata(dataset, traindata, testdata):
    w1, w0 = coefficients(traindata)
    x_data = [x[0] for x in dataset]
    x_max = max(x_data)
    y_data = [y[-1] for y in dataset]
    y_max = max(y_data)
    x = x_data
    y = [t*w1+w0 for t in x]
    plt.axis([0, x_max, 0, y_max])
    plt.plot(x_data, y_data, 'bs')
    plt.plot(x, y)
    plt.grid()
    plt.show()


# 测试
dataset = load_csv('file.csv')
print(dataset)
str_to_float(dataset)
print(dataset)
percent = 0.6
rmse, traindata, testdata = evaluate_algorithm(dataset, linear_regression, percent)
drawdata(dataset, traindata, testdata)
print("RMSE:%.3f" % (rmse))