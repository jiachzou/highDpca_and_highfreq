# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 17:09:02 2017

@author: Jiacheng Z
"""
#----this is a wrapper for code in question 2----
import os
import re

import factorSimulation
import vanillaPCA
import risk_premiumPCA
import numpy as np
from numpy.linalg import inv as invert
from datetime import datetime

def NOW():
    return str(datetime.now())[:-7]

def NOWDIGIT():
    return re.sub(pattern='[-: ]*', repl="", string=NOW())

def readTrue():
    trueFactors = dict()
    trueLoadings = dict()
    factorDir = r"C:\Users\Jiacheng Z\Dropbox\Courses\17Spring\MS&E349\MS&E349_Shared\HW2\code"

    for idx in range(1,4):
        CollectionKey = 'dataset' + str(idx)

        FactorFilepath = os.path.join(factorDir,'Simulation_Factor_{0}.csv'.format(idx))
        trueFactors[CollectionKey] = np.genfromtxt(FactorFilepath, delimiter=',')

        LoadingFilepath = os.path.join(factorDir,'Simulation_Loading_{0}.csv'.format(idx))
        trueLoadings[CollectionKey] = np.genfromtxt(LoadingFilepath, delimiter=',')

    return trueFactors, trueLoadings

def maxSharpe(Factors):
    assert Factors.shape == (150,3),'[Error] Factor dimensionality error!\n'
    MuVec = np.mean(Factors, axis=0).reshape((3,1))
    Covariance = np.cov(Factors,rowvar=False)
    MaxSharpe = MuVec.T.dot(invert(Covariance)).dot(MuVec)
    return MaxSharpe

def main(reSimulate=True):
    if reSimulate:  factorSimulation.main()
    vanillaFactors, vanillaLoadings = vanillaPCA.main()
    rpFactors, rpLoadings = risk_premiumPCA.main()
    trueFactors, trueLoadings = readTrue()

    #----------part 6: maximum Sharpe-ratio---------
    MaxSharpes = dict()
    for idx in range(1, 4):
        CollectionKey = 'dataset' + str(idx)
        MaxSharpes['True'] = maxSharpe(trueFactors[CollectionKey])
        MaxSharpes['VanillaPCA'] = maxSharpe(vanillaFactors[CollectionKey])
        MaxSharpes['RPPCA'] = maxSharpe(rpFactors[CollectionKey])
        print('{0}\n[INFO] Finished calculated Max Sharpe ratios for dataset {1}.\n * True Factors: {2};\n * Vanilla PCA: {3};\n * RP-PCA: {4};'.
              format('=' * 20 + NOW() + '=' * 20, idx, MaxSharpes['True'], MaxSharpes['VanillaPCA'], MaxSharpes['RPPCA']))


if __name__ == '__main__':
    main()


