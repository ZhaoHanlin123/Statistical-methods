import numpy as np


observation = ['red', 'write']  # 所有可能的状态观测集合


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.M = B.shape[1]
        self.N = A.shape[0]


def backward(hmm, obs):
    T = len(obs)  # 观察序列长度
    N = hmm.N  # 隐藏层状态数
    beta = np.matrix(np.zeros((N, T)))  # 后向概率
    beta[:, T - 1] = 1
    for t in reversed(range(0, T-1)):
        for n in range(0, N):
            beta[n, t] = sum(np.multiply(np.multiply(hmm.A[n, :].T, hmm.B[:, observation.index(obs[t+1])]),
                                         beta[:, t+1]))
    prob = sum(np.multiply(np.multiply(hmm.pi, hmm.B[:, observation.index(obs[0])]), beta[:, 0]))
    return prob, beta


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
    prob, alpha = backward(hmm, observed)
    print("The backword probability is: \n", alpha)
    print("The probability of this observed sequence is %f" % prob)
