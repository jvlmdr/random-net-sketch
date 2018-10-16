from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def main():
    L = 5
    min_m, max_m = 100, 1000
    m = list(reversed(sorted(np.random.randint(min_m, max_m, L + 1))))
    # m[0], m[L] = max_m, min_m

    alpha_w = {}
    for i in reversed(range(1, L + 1)):
        # alpha_w[i] = 1 / (m[i - 1] * m[i]) ** (1 / 2)
        # alpha_w[i] = 1 / (m[i - 1] * m[i]) ** (1 / 4)
        alpha_w[i] = np.sqrt(2 / np.sqrt(m[i - 1] * m[i]))
        # alpha_w[i] = 1 if i == L else np.sqrt(m[i + 1] / m[i - 1]) * alpha_w[i + 1]
        # alpha_w[i] = np.sqrt(2 / m[0] * m[L - 1] / m[1]) if i == 1 else np.sqrt(2 / m[i])
        # alpha_w[i] = np.sqrt(2 / m[i])
        print('layer {}, alpha[i] {:8.3f}'.format(i, alpha_w[i]))

    # A = np.zeros((L, L))
    # # A = np.zeros((L, L + 1))
    # q = np.zeros((L,))
    # for i in range(L):
    #     q[i] = (L - 2 * i + 1) * np.log(2)
    #     for j in range(L):
    #         if j <= i:
    #             A[i][j] = -1
    #             q[i] += np.log(m[j])
    #         elif j >= i + 1:
    #             A[i][j] = 1
    #             q[i] -= np.log(m[j + 1])
    #     # A[i][L] = -1
    # print(A)
    # print(q)
    # v, _, _, _ = np.linalg.lstsq(A, q, rcond=None)
    # print(v)
    # for i in range(L):
    #     alpha_w[i + 1] = np.sqrt(np.exp(v[i]))

    x = {0: np.random.randn(m[0])}
    y = {}
    w = {}
    for i in range(1, L + 1):
        print('layer {}, alpha[i] {:8.3f}'.format(i, alpha_w[i]))
        w[i] = alpha_w[i] * np.random.randn(m[i], m[i - 1])
    # b = {i: np.random.randn(m[i]) for i in range(1, L + 1)}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    for i in range(1, L + 1):
        y[i] = np.dot(w[i], x[i - 1]) + b[i]
        if i < L:
            x[i] = relu(y[i])
        print('layer {}, std x[i-1] {:8.3f}, std y[i] {:8.3f}'.format(
            i, np.std(x[i - 1]), np.std(y[i])))

    ddy = {L: np.random.randn(m[L])}
    ddx = {}
    ddw = {}
    ddb = {}
    for i in reversed(range(1, L + 1)):
        if i < L:
            ddx[i] = np.dot(np.transpose(w[i + 1]), ddy[i + 1])
            ddy[i] = np.multiply(relu_gradient(y[i]), ddx[i])
            assert ddx[i].shape == x[i].shape
            assert ddy[i].shape == y[i].shape
        ddw[i] = np.multiply(ddy[i][:, None], x[i-1][None, :])
        ddb[i] = ddy[i]
        print('layer {}, std ddw[i] {:8.3f} (rel {:6.3f}), std ddy[i] {:8.3f} (rel {:6.3f}), rel/rel {:6.3f}'.format(
            i,
            np.std(ddw[i]), np.std(ddw[i]) / np.std(w[i]),
            np.std(ddy[i]), np.std(ddy[i]) / np.std(y[i]),
            (np.std(ddw[i]) / np.std(w[i])) / (np.std(ddy[i]) / np.std(y[i]))))

    for i in reversed(range(1, L + 1)):
        print('layer {}, ddw[i]/w[i] / ddy[i]/y[i] {:8.3f}'.format(
            i, (np.std(ddw[i]) / np.std(w[i])) / (np.std(ddy[i]) / np.std(y[i]))))
    print('ddw/w = about {:6.3f}'.format(np.sqrt(0.5 * np.sqrt(m[0] * m[L]))))
    print('ddy/y = about {:6.3f}'.format(np.sqrt(0.5 * np.sqrt(m[L] / m[0]))))
    print('(ddw/w)/(ddy/y) = about {:6.3f}'.format(np.sqrt(m[0])))

def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))

def relu_gradient(x):
    return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

if __name__ == '__main__':
    main()
