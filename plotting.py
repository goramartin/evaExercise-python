# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 
import sys
import functools


def add_suffix(str, suff):
    return str + '.' + suff

fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
datasets = ['default', 'P1UniformCX', 'P1ArithmeticCX', 'P1AdaptiveOneFifthMut', 'P1AdaptiveOneFifthMut+UniformCX', 'P1AdaptiveOneFifthMut+ArithmeticCX', 'P2DifferentialStaticCR09F055', 'P2DifferentialAdaptiveCR09-F035-FD035-CRD09', 'P3Baldwin+MUT0.8+CX0.3+UniformCX', 'P3Baldwin+Depth10+MUT0.8+CX0.3+UniformCX']
fit_name = fit_names[int(sys.argv[1])]
f = functools.partial(add_suffix, suff=fit_name)

plt.figure(figsize=(12,8))
utils.plot_experiments('continuous', list(map(f, datasets)), {'P3Baldwin+MUT0.8+CX0.3+UniformCX': 'P3Lamarck+MUT0.8+CX0.3+UniformCX'})
plt.yscale('log')
plt.show()
 