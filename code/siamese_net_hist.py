from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

BRANCHES = ['A', 'B']

cmap = plt.get_cmap('plasma')

def main():
    L = 5

    for size_ind in range(4):
        # Fix the dimension and sample several parameter vectors.
        # For each parameter vector, evaluate a batch of random inputs.
        plt.clf()
        m = 100 * np.random.randint(5, 20, L + 1)
        print('m:', list(enumerate(m)))
        for param_ind in range(5):
            y_out = evaluate(m, B=200)
            plt.hist(y_out)
        plt.xlim(-40, 40)
        plt.savefig('siamese_hist_L_{}_size_{}.pdf'.format(L, size_ind + 1))

    plt.clf()
    y_out = []
    for param_ind in range(100):
        # For each network, pick a random dimension.
        # For each parameter vector, evaluate a batch of random inputs.
        m = 100 * np.random.randint(5, 20, L + 1)
        y_out.append(evaluate(m, B=100))
        plt.hist(y_out[-1])
    plt.savefig('siamese_hist_L_{}.pdf'.format(L))

    plt.clf()
    plt.hist(np.ravel(y_out))
    plt.savefig('siamese_hist_L_{}_ravel.pdf'.format(L))


def evaluate(m, B=32, color=None):
    L = len(m) - 1
    alpha_w = {i: np.sqrt(2 / m[i]) for i in range(1, L + 1)}
    for i in range(1, L + 1):
        print('layer {}, alpha[i] {:8.3f}'.format(i, alpha_w[i]))
    w_out = np.sqrt(1 / (4 * m[L]))
    b_out = -np.sqrt(m[L])

    x = {s: {0: np.random.randn(B, m[0])} for s in BRANCHES}
    y = {s: {} for s in BRANCHES}
    w = {i: alpha_w[i] * np.random.randn(m[i], m[i - 1]) for i in range(1, L + 1)}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    for i in range(1, L + 1):
        for s in BRANCHES:
            y[s][i] = np.dot(x[s][i - 1], np.transpose(w[i])) + b[i]
            if i < L:
                x[s][i] = relu(y[s][i])
            print('branch {}, layer {}, std x[i-1] {:8.3f}, std y[i] {:8.3f}'.format(
                s, i, np.std(x[s][i - 1]), np.std(y[s][i])))
    print('mean(y_L) {:8.3f}, {:8.3f}'.format(np.mean(y['A'][L]), np.mean(y['B'][L])))
    print('var(y_L) {:8.3f}, {:8.3f}'.format(np.var(y['A'][L]), np.var(y['B'][L])))
    u = inner_prod(y['A'][L], y['B'][L], axis=-1, keepdims=True)
    print('u {:8.3f}'.format(np.mean(u)))
    y_out = w_out * u + b_out
    print('y_out {:8.3f}'.format(np.mean(y_out)))

    return y_out


def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))

def relu_gradient(x):
    return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

def other(branch):
    assert branch in BRANCHES
    if branch == 'A':
        return 'B'
    else:
        return 'A'

def inner_prod(x, y, **kwargs):
    return np.sum(np.multiply(x, y), **kwargs)

if __name__ == '__main__':
    main()
