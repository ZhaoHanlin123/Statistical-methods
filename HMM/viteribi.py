import numpy as np
observation = ['red', 'write']


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.M = B.shape[1]
        self.N = A.shape[0]


def viterbi(hmm, obs):
    T = len(obs)
    N = hmm.N
    psi = np.zeros((N, T))  # reverse pointer
    delta = np.zeros((N, T))
    q = np.zeros(T)

    delta[:, 0] = hmm.pi[:] * hmm.B[:, observation.index(obs[0])]  # 初始化
    # 动态规划过程
    for t in range(1, T):
        for n in range(0, N):
            temp = delta[:, t-1] * hmm.A[:, n]
            max_ind = np.argmax(temp)
            psi[n, t] = max_ind
            delta[n, t] = hmm.B[n, observation.index(obs[t])] * temp[max_ind]

    # 最优路径回溯
    max_ind = np.argmax(delta[:, T-1])
    q[T-1] = max_ind
    prob = delta[:, T-1][max_ind]
    for t in reversed(range(0, T-1)):
        q[t] = psi[int(q[t+1]), t+1]

    return prob, q, delta


if __name__ == "__main__":
    A = np.array([[0.5, 0.2, 0.3],
               [0.3, 0.5, 0.2],
               [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
               [0.4, 0.6],
               [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])

    hmm = HMM(pi, A, B)

    observed = ['red', 'write', 'red']
    prob, q, delta = viterbi(hmm, observed)

    print('The maximum probability of every single path is:\n', delta)
    print('The probability of optimal path is %f.' % prob)
    print('The optimal sequence of state is:\n', q)
