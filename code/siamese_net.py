from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

BRANCHES = ['A', 'B']

def main():
    L = 5
    m = np.random.randint(500, 2000, L + 1)
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

    x = {s: {0: np.random.randn(m[0])} for s in BRANCHES}
    y = {s: {} for s in BRANCHES}
    w = {i: alpha_w[i] * np.random.randn(m[i], m[i - 1]) for i in range(1, L + 1)}
    b = {i: np.zeros(m[i]) for i in range(1, L + 1)}
    rho = 1
    # w_out u + b_out = (w_out / rho) (rho u) + b_out
    # w_out = np.sqrt(1 / (4 * m[L]))
    w_out = (1 / rho) * np.sqrt(1 / (4 * m[L]))
    b_out = -np.sqrt(m[L])
    # b_out = 0
    for i in range(1, L + 1):
        for s in BRANCHES:
            y[s][i] = np.dot(w[i], x[s][i - 1]) + b[i]
            if i < L:
                x[s][i] = relu(y[s][i])
            print('branch {}, layer {}, std x[i-1] {:8.3f}, std y[i] {:8.3f}'.format(
                s, i, np.std(x[s][i - 1]), np.std(y[s][i])))
    print('mean(y_L) {:8.3f}, {:8.3f}'.format(np.mean(y['A'][L]), np.mean(y['B'][L])))
    print('var(y_L) {:8.3f}, {:8.3f}'.format(np.var(y['A'][L]), np.var(y['B'][L])))
    u = np.tensordot(y['A'][L], y['B'][L], axes=1)
    print('u {:8.3f}'.format(u))
    print('rho * u {:8.3f}'.format(rho * u))
    y_out = w_out * rho * u + b_out
    print('y_out {:8.3f}'.format(y_out))

    # for i in range(1, L + 1):
    #     print('layer {}, E(y^A * y^B) = {:7.3f}, E(y^A * y^B) = {:7.3f}, var(y^A) = {:7.3f}'.format(
    #         i, np.mean(y['A'][i] * y['B'][i]), np.mean(y['A'][i]) * np.mean(y['B'][i]), np.var(y['A'][i])))
    # return

    ddy_out = 1.0
    ddw_out = ddy_out * np.tensordot(y['A'][L], y['B'][L], axes=1)
    ddb_out = ddy_out
    print('abs ddw_out {:8.3f}, abs ddb_out {:8.3f}'.format(np.abs(ddw_out), np.abs(ddb_out)))
    ddy = {s: {L: ddy_out * w_out * y[other(s)][L]} for s in BRANCHES}
    for s in BRANCHES:
        print('branch {}, std ddy[L] {:8.3f}'.format(s, np.std(ddy[s][L])))
    ddx = {s: {} for s in BRANCHES}
    for i in reversed(range(1, L)):
        for s in BRANCHES:
            ddx[s][i] = np.dot(np.transpose(w[i + 1]), ddy[s][i + 1])
            ddy[s][i] = np.multiply(relu_gradient(y[s][i]), ddx[s][i])
            assert ddx[s][i].shape == x[s][i].shape
            assert ddy[s][i].shape == y[s][i].shape
            print('branch {}, layer {}, std ddy[s][i] {:8.3f}, std ddx[s][i] {:8.3f}'.format(
                s, i, np.std(ddy[s][i]), np.std(ddx[s][i])))

    ddw = {i: 0 for i in range(1, L + 1)}
    ddb = {i: 0 for i in range(1, L + 1)}
    for i in reversed(range(1, L + 1)):
        for s in BRANCHES:
            ddw[i] += np.multiply(ddy[s][i][:, None], x[s][i-1][None, :])
            ddb[i] += ddy[s][i]
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

if __name__ == '__main__':
    main()
