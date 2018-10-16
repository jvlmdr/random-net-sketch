from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def main():
    L = 5
    m = list(reversed(sorted(np.random.randint(300, 3000, L + 1))))

    s = {1: m[0] ** -0.25}
    for i in range(1, L):
        # a[i + 1] = a[i] * np.sqrt(m[i - 1] / m[i + 1])
        s[i + 1] = s[i] * np.sqrt(m[i - 1] / m[i])
    a = {i: (1 / s[i]) * np.sqrt(2 / m[i - 1]) for i in range(1, L + 1)}

    u = {}
    for i in range(1, L + 1):
        print('layer {}, s[i] {:8.3f}, a[i] {:8.3f}'.format(i, s[i], a[i]))
        u[i] = s[i] * np.random.randn(m[i], m[i - 1])

    x = {0: np.random.randn(m[0])}
    y = {}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    for i in range(1, L + 1):
        y[i] = a[i] * np.dot(u[i], x[i - 1]) + b[i]
        if i < L:
            x[i] = relu(y[i])
        print('layer {}, std x[i-1] {:8.3f}, std y[i] {:8.3f}'.format(i, np.std(x[i - 1]), np.std(y[i])))

    ddy = {L: np.random.randn(m[L])}
    print('layer {}, std ddy[i] {:8.3f}'.format(L, np.std(ddy[L])))
    ddx = {}
    ddu = {}
    ddb = {}
    for i in reversed(range(1, L + 1)):
        if i < L:
            ddx[i] = a[i + 1] * np.dot(np.transpose(u[i + 1]), ddy[i + 1])
            ddy[i] = np.multiply(relu_gradient(y[i]), ddx[i])
            print('layer {}, std ddy[i] {:8.3f}, std ddx[i] {:8.3f}'.format(i, np.std(ddy[i]), np.std(ddx[i])))
            assert ddx[i].shape == x[i].shape
            assert ddy[i].shape == y[i].shape
        ddu[i] = a[i] * np.multiply(ddy[i][:, None], x[i-1][None, :])
        ddb[i] = ddy[i]

    for i in reversed(range(1, L + 1)):
        print('layer {}, std ddu[i] {:8.3f}, std ddb[i] {:8.3f}'.format(i, np.std(ddu[i]), np.std(ddb[i])))

def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))

def relu_gradient(x):
    return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

def geomean(x):
    return np.exp(np.mean(np.log(x)))

if __name__ == '__main__':
    main()
