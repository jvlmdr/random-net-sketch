from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def main():
    n = int(1e7)

    mu, sigma = 0, 1
    y = sigma * np.random.randn(n) + mu
    x = relu(y)
    print('mean {:.3f}, var {:.3f}, mean sqr {:.3f}'.format(np.mean(y), np.var(y), np.mean(y ** 2)))
    print('mean {:.3f}, var {:.3f}, mean sqr {:.3f}'.format(np.mean(x), np.var(x), np.mean(x ** 2)))

    # var[y] = 1/12 (2 r)^2 = 1/3 r^2 = 1; r^3 = 3
    r = np.sqrt(3)
    y = np.random.uniform(-r, r, n)
    x = relu(y)
    print('mean {:.3f}, var {:.3f}, mean sqr {:.3f}'.format(np.mean(y), np.var(y), np.mean(y ** 2)))
    print('mean {:.3f}, var {:.3f}, mean sqr {:.3f}'.format(np.mean(x), np.var(x), np.mean(x ** 2)))


def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))


if __name__ == '__main__':
    main()
