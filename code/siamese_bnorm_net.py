from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

BRANCHES = ['A', 'B']

def main():
    L = 5
    B = 40
    m = np.random.randint(200, 800, L + 1)
    # m = list(reversed(sorted(np.random.randint(100, 1000, L + 1))))
    print('m:', list(enumerate(m)))

    alpha_w = {i: np.sqrt(2 / m[i]) for i in range(1, L + 1)}
    # alpha_w = {}
    # for i in reversed(range(1, L + 1)):
    #     # alpha_w[i] = 1
    #     # alpha_w[i] = 1 if i == L else np.sqrt(m[i + 1] / m[i - 1]) * alpha_w[i + 1]
    #     alpha_w[i] = np.sqrt(2 / m[0] * m[L - 1] / m[1]) if i == 1 else np.sqrt(2 / m[i])
    #     # alpha_w[i] = np.sqrt(m[L] / (m[0] * m[1])) if i == 1 else np.sqrt(2 / m[i])
    #     # alpha_w[i] = np.sqrt(2 / m[i])
    # alpha_w[L + 1] = np.sqrt(1 / (4 * m[L]))
    for i in range(1, L + 1):
        print('layer {}, alpha[i] {:8.3f}'.format(i, alpha_w[i]))

    x = {s: {0: np.random.randn(B, m[0])} for s in BRANCHES}
    y = {s: {} for s in BRANCHES}
    w = {i: alpha_w[i] * np.random.randn(m[i], m[i - 1]) for i in range(1, L + 1)}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    rho = 1
    # w_out u + b_out = (w_out / rho) (rho u) + b_out
    # w_out = np.sqrt(1 / (4 * m[L]))
    # w_out = (1 / rho) * np.sqrt(1 / (4 * m[L]))
    # b_out = -np.sqrt(m[L])
    w_out = 1
    b_out = 0
    for i in range(1, L + 1):
        for s in BRANCHES:
            y[s][i] = np.dot(x[s][i - 1], np.transpose(w[i])) + b[i]
            if i < L:
                x[s][i] = relu(y[s][i])
            print('branch {}, layer {}, std x[i-1] {:8.3f}, std y[i] {:8.3f}'.format(
                s, i, np.std(x[s][i - 1]), np.std(y[s][i])))
    print('mean(y_L) {:8.3f}, {:8.3f}'.format(np.mean(y['A'][L]), np.mean(y['B'][L])))
    print('var(y_L) {:8.3f}, {:8.3f}'.format(np.var(y['A'][L]), np.var(y['B'][L])))
    u = inner_prod(y['A'][L], y['B'][L])
    print('u {:8.3f}'.format(np.mean(u)))
    q = bnorm(u)
    y_out = w_out * rho * q + b_out
    print('mean y_out {:8.3f}, std y_out {:8.3f}'.format(np.mean(y_out), np.std(y_out)))

    ddy_out = 1 / B * np.random.randn(B, 1)
    ddq = ddy_out * rho * w_out
    ddu = grad_bnorm(ddq, u)
    ddw_out = ddy_out * rho * q
    ddb_out = ddy_out
    print('abs ddw_out {:8.3f}, abs ddb_out {:8.3f}'.format(
        np.mean(np.abs(ddw_out)), np.mean(np.abs(ddb_out))))
    ddy = {s: {L: ddu * y[other(s)][L]} for s in BRANCHES}
    for s in BRANCHES:
        print('branch {}, std ddy[L] {:8.3f}'.format(s, np.std(ddy[s][L])))
    ddx = {s: {} for s in BRANCHES}
    for i in reversed(range(1, L)):
        for s in BRANCHES:
            ddx[s][i] = np.dot(ddy[s][i + 1], w[i + 1])
            ddy[s][i] = np.multiply(relu_gradient(y[s][i]), ddx[s][i])
            assert ddx[s][i].shape == x[s][i].shape
            assert ddy[s][i].shape == y[s][i].shape
            print('branch {}, layer {}, std ddy[s][i] {:8.3f}, std ddx[s][i] {:8.3f}'.format(
                s, i, np.std(ddy[s][i]), np.std(ddx[s][i])))

    ddw = {i: 0 for i in range(1, L + 1)}
    ddb = {i: 0 for i in range(1, L + 1)}
    for i in reversed(range(1, L + 1)):
        for s in BRANCHES:
            ddw[i] += np.sum(np.multiply(np.expand_dims(ddy[s][i], axis=-1),
                                         np.expand_dims(x[s][i-1], axis=-2)), axis=0)
            ddb[i] += np.sum(ddy[s][i], axis=0)
        print('layer {}, std ddw[i] {:8.3f}, std ddb[i] {:8.3f}'.format(
            i, np.std(ddw[i]), np.std(ddb[i])))
        # print('layer {}, std ddw[i] {:8.3f}, std ddb[i] {:8.3f}'.format(
        #     i, np.std(ddw[i]) / np.std(w[i]), np.std(ddb[i]) / np.std(y[s][i])))

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

def bnorm(y, eps=None):
    u = bnorm_mean(y)
    q = bnorm_var(u, eps=eps)
    return q

def bnorm_mean(y):
    c = np.mean(y, axis=0, keepdims=True)
    u = y - c
    return u

def bnorm_var(u, eps=None):
    if eps is None:
        eps = 1e-4
    v = u ** 2
    s = np.mean(v, axis=0, keepdims=True)
    r = np.sqrt(s)
    a = 1 / (r + eps)
    q = np.multiply(a, u)
    return q

def grad_bnorm(ddq, y, eps=None):
    u = bnorm_mean(y)
    ddu = grad_bnorm_var(ddq, u, eps=eps)
    ddy = grad_bnorm_mean(ddu, y)
    return ddy

def grad_bnorm_mean(ddu, y):
    ddy = ddu - np.mean(ddu, axis=0, keepdims=True)
    return ddy

def grad_bnorm_var(ddq, u, eps=None):
    if eps is None:
        eps = 1e-4
    v = u ** 2
    s = np.mean(v, axis=0, keepdims=True)
    r = np.sqrt(s)
    a = 1 / (r + eps)
    q = np.multiply(a, u)
    ddu = np.multiply(a, ddq - np.multiply(np.mean(ddq * q, axis=0, keepdims=True), q))
    return ddu

if __name__ == '__main__':
    main()
