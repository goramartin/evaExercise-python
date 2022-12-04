# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 
import sys
import functools


def add_suffix(str, suff):
    return str + '.' + suff

fit_names = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']
datasets = ['default', 'default+MS0.5', 'CXUniform+DefMut+CXP0.2+MP0.8+MS0.5', 'CXUniform+DefMut+CXP0.8+MP0.2+MS0.5', 'CXArithmetics+DefMut+CXP0.8+MP0.2+MS0.5', 'CXArithmetics+DefMut+CXP0.2+MP0.8+MS0.5', 'DefCX+AdaptiveMut+CXP0.8+MP0.2+MS0.5', 'DefCX+AdaptiveMut+CXP0.8+MP0.2+MS0.05', 'CXUniform+AdaptiveMut+CXP0.8+MP0.2+MS0.5+sum', 'CXUniform+DiffMut+CXP0.8+MP0.2+MS0.5']

for name in fit_names:
    f = functools.partial(add_suffix, suff=name)
    plt.figure(figsize=(12,8))
    utils.plot_experiments('multi', list(map(f, datasets)), { 'DefCX+AdaptiveMut+CXP0.8+MP0.2+MS0.5': 'CXUniform+AdaptiveMut+CXP0.8+MP0.2+MS0.5',  'DefCX+AdaptiveMut+CXP0.8+MP0.2+MS0.05': 'CXUniform+AdaptiveMut+CXP0.8+MP0.2+MS0.05' })
    #plt.yscale('log')
    plt.show()
 