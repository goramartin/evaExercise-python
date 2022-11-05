# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 

plt.figure(figsize=(12,8))
utils.plot_experiments('partition', ['default', 'hw2-diffavg-tournament-popsize100-maxgen500-mutprob1-cxprob0--elite0.07', 'hw2-diffavg-tournament-popsize100-maxgen500-mutprob0.2-cxprob0.8--elite0.07', 'hw3-diffavg-tournament-swapmaxmin-popsize100-maxgen500-mutprob1-cxprob0.8--elite0.07', 'hw3-diffavg-tournament-swapmaxminbest-popsize100-maxgen500-mutprob1-cxprob0.4--elite0.07'])
plt.show()
 