from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

BRANCHES = ['A', 'B']

cmap = plt.get_cmap('plasma')

def main():
    L = 5

    for method in ['independent', 'equal', 'compromise']:
        for size_ind in range(4):
            # Fix the dimension and sample several parameter vectors.
            # For each parameter vector, evaluate a batch of random inputs.
            plt.clf()
            m = np.random.randint(200, 1000, L + 1)
            m[L] = int((size_ind + 1) / 4 * 1000)
            print('m:', list(enumerate(m)))
            for param_ind in range(5):
                y_out = evaluate(m, B=200, output_method=method)
                plt.hist(y_out, alpha=0.5)
            plt.xlim((-30, 50))
            plt.savefig('siamese_hist_{}_size_{}.pdf'.format(method, size_ind + 1))

    for method in ['independent', 'equal', 'compromise']:
        plt.clf()
        y_out = []
        for param_ind in range(100):
            # For each network, pick a random dimension.
            # For each parameter vector, evaluate a batch of random inputs.
            m = 100 * np.random.randint(5, 20, L + 1)
            # print('m:', list(enumerate(m)))
            y_out.append(evaluate(m, B=100, output_method=method))
            plt.hist(y_out[-1], alpha=0.5)
        plt.savefig('siamese_hist_{}.pdf'.format(method))
        plt.clf()
        plt.hist(np.ravel(y_out))
        plt.savefig('siamese_hist_{}_ravel.pdf'.format(method))


def evaluate(m, B=32, output_method='independent'):
    L = len(m) - 1
    alpha_w = {i: np.sqrt(2 / m[i - 1]) for i in range(1, L + 1)}

    # E[u] = 0 if embeddings are independent, 2 m[L] if equal
    # V[u] = m_L V[y_L]^2 if independent, 2 m_L V[y_L]^2 if equal
    # => w_out = 1/sqrt(2 m_L) if independent, 1/sqrt(4 m_L) if equal
    if output_method == 'independent':
        w_out = 1 / np.sqrt(2 * m[L])
        b_out = 0
    elif output_method == 'equal':
        w_out = 1 / np.sqrt(4 * m[L])
        b_out = -2 * m[L] * w_out
    elif output_method == 'compromise':
        w_out = 1 / np.sqrt(np.sqrt(8) * m[L])
        b_out = -m[L] * w_out

    x = {s: {0: np.random.randn(B, m[0])} for s in BRANCHES}
    # x0 = np.random.randn(B, m[0])
    # x = {s: {0: np.array(x0)} for s in BRANCHES}
    y = {s: {} for s in BRANCHES}
    w = {i: alpha_w[i] * np.random.randn(m[i], m[i - 1]) for i in range(1, L + 1)}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    for i in range(1, L + 1):
        for s in BRANCHES:
            y[s][i] = np.dot(x[s][i - 1], np.transpose(w[i])) + b[i]
            if i < L:
                x[s][i] = relu(y[s][i])
    u = inner_prod(y['A'][L], y['B'][L])
    y_out = w_out * u + b_out
    print('var_y_L {:8.3f}, y_out {:8.3f}'.format(np.var(y['A'][L]), np.mean(y_out)))

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

def inner_prod(x, y, axis=-1, keepdims=True):
    return np.sum(np.multiply(x, y), axis=axis, keepdims=keepdims)

if __name__ == '__main__':
    main()
