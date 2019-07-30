import numpy as np

class bayes:
    def __init__(self, lamb=1):
        if (lamb < 0):  # 因为lamb大于等于0
            self._lamb = 1
        else:
            self._lamb = lamb

    # 创建词汇表
    def create_vocab_list(self, dataSet):
        vocab_set = set([])
        m, n = np.shape(dataSet)  # 获得数据集的行和列
        self._S = []  # 每个特征在自己的维度里出现的次数
        for i in range(n):  # 按照维度来创建词条，第一维，第二维
            column = set([row[i] for row in dataSet])
            vocab_set = vocab_set | set(column)
            self._S.extend(list(np.ones(len(column)) * len(column)))
        return list(vocab_set)

    def set_of_word2vec(self, vocab_list, input_set):
        vocab_vec = [0] * len(vocab_list)  # vocablist大小的零向量 zero vector
        for word in input_set:  # 遍历输入样本的每个特征
            if word in vocab_list:
                vocab_vec[vocab_list.index(word)] = 1  # 如果发现有匹配的值就设置为1
        return vocab_vec

    # 训练样本
    def train(self, dataSet, labels):
        self._vocab_list = self.create_vocab_list(dataSet)  # 创建特征词汇表 create vocal list
        train_matrix = []  # 多条词条向量的矩阵（一个词条向量代表着一个样本在词条中出现的次数）
        for line in dataSet:  # 将每个训练样本转换为词条向量
            train_matrix.append(self.set_of_word2vec(self._vocab_list, line))
        n = len(self.vocab_list)  # 词条的特征数
        print(n)
        negative_feature_num = np.zeros(n)  # 在类别为-1时，出现特征的次数向量
        positive_feature_num = np.zeros(n)  # 在类别为 1时，出现特征的次数向量
        negative_num = 0  # 标签中出现-1的次数
        positive_num = 0  # 标签中出现 1的次数
        for i in range(len(train_matrix)):
            if labels[i] == 1:
                positive_num += 1
                positive_feature_num += train_matrix[i]
            else:
                negative_num += 1
                negative_feature_num += train_matrix[i]
        self._positive_vec = (positive_feature_num + self.lamb) / (
                                positive_num + np.array(self.S) * self.lamb)  # p1的各个随机向量（特征）的概率分布
        self._negative_vec = (negative_feature_num + self.lamb) / (
                                negative_num + np.array(self.S) * self.lamb)
        self._p_positive = (positive_num + self.lamb) / float(len(labels) + len(set(labels)) * self.lamb)  # p(y=1)的概率

    def predict(self, input_data):  # 预测函数
        input_vec = self.set_of_word2vec(self.vocab_list, input_data)  # 测试样本的词条向量
        p_positive = self.p_positive  # 按照公式需要乘以p(y=1)的值，我们就以此为初始值
        p_negative = 1 - self.p_positive
        for num in np.multiply(self.positive_vec, input_vec):  # 概率分布和词条向量进行相乘，得出p(x=xi|y=1)的概率
            if (num > 0):
                p_positive *= num
        for num in np.multiply(self.negative_vec, input_vec):
            if(num > 0):
                p_negative *= num
        print(p_positive, p_negative)
        if (p_positive > p_negative):  # 相比，谁大就倾向谁 up to max probability
            return 1
        else:
            return -1

    @property
    def lamb(self):
        return self._lamb

    @property
    def vocab_list(self):
        return self._vocab_list

    @property
    def positive_vec(self):
        return self._positive_vec

    @property
    def negative_vec(self):
        return self._negative_vec

    @property
    def p_positive(self):
        return self._p_positive

    @property
    def S(self):
        return self._S


if __name__ == "__main__":
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    bayes = bayes()
    bayes.train(dataSet, labels)
    print("prediction is:", bayes.predict([2, "S"]))

