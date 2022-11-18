# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 
import sys
import functools


def add_suffix(str, suff):
    return str + '.' + suff

datasets = ['default', 'P1UniformCX', 'P1ArithmeticCX', 'P1AdaptiveOneFifthMut', 'P1AdaptiveOneFifthMut+UniformCX', 'P1AdaptiveOneFifthMut+ArithmeticCX']
fit_name = sys.argv[1]
f = functools.partial(add_suffix, suff=fit_name)

plt.figure(figsize=(12,8))
utils.plot_experiments('continuous', list(map(f, datasets)))
plt.yscale('log')
plt.show()
 