import numpy as np

observation = ['red', 'write']  # 所有可能的状态观测集合


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.M = B.shape[1]
        self.N = A.shape[0]


def forward(hmm, obs):
    T = len(obs)  # 观察序列的长度
    N = hmm.N  # 隐藏层状态数
    alpha = np.matrix(np.zeros((N, T)))  # 前向概率
    alpha[:, 0] = np.multiply(hmm.pi[:], hmm.B[:, observation.index(obs[0])])  # 初值
    for t in range(1, T):  # 递推
        for n in range(0, N):
            alpha[n, t] = sum(alpha[:, t-1].T * hmm.A[:, n]) * hmm.B[n, observation.index(obs[t])]
    prob = sum(alpha[:, T-1])  # 计算观察序列概率
    return prob, alpha


if __name__ == "__main__":
    # 状态转移概率矩阵
    A = np.matrix([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])

    # 观测概率矩阵
    B = np.matrix([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    pi = np.matrix([0.2, 0.4, 0.4]).T
    hmm = HMM(pi, A, B)
    observed = ['red', 'write', 'red']  # 观测序列
    prob, alpha = forward(hmm, observed)
    print("The forword probability is: \n", alpha)
    print("The probability of this observed sequence is %f" % prob)
