# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 20:29:07 2017

@author: Jiacheng Z
"""
#----this correspond to question 4----
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from run import NOW
import pandas as pd

def main():
    N_WeeklyObs = 385

    file_loc = 'https://raw.githubusercontent.com/jiacheng0409/pca/master/HF_Data.csv'
    rwData = pd.read_csv(file_loc)

    SPY = rwData.iloc[:,0].as_matrix()
    DeltaSPY = np.diff(SPY, n=1)
    Sq_DeltaSPY = np.square(DeltaSPY)
    VolLenSPY = len(SPY)//N_WeeklyObs +1
    VolSPY = np.array([0.0]*VolLenSPY)

    AAPL = rwData.iloc[:, 1].as_matrix()
    DeltaAAPL = np.diff(AAPL, n=1)
    Sq_DeltaAAPL = np.square(DeltaAAPL)
    VolAAPL = np.array([0.0]*VolLenSPY)

    for idx in range(VolLenSPY):
        if idx+N_WeeklyObs>len(SPY):
            WeeklyChunkSPY = Sq_DeltaSPY[idx*N_WeeklyObs:]
            WeeklyChunkAAPL = Sq_DeltaAAPL[idx*N_WeeklyObs:]
        else:
            WeeklyChunkSPY = Sq_DeltaSPY[idx*N_WeeklyObs:(idx + 1)*N_WeeklyObs]
            WeeklyChunkAAPL = Sq_DeltaAAPL[idx*N_WeeklyObs:(idx + 1)*N_WeeklyObs]

        VolSPY[idx] = np.sum(WeeklyChunkSPY)
        VolAAPL[idx] = np.sum(WeeklyChunkAAPL)

    plt.figure()
    plt.plot(VolAAPL,label='AAPL High-frequency Volatility')
    plt.plot(VolSPY, label='SPY High-frequency Volatility')
    plt.title('Fig.7 Weeekly Volatility')
    plt.legend()
    plt.savefig('Fig_7.png')

    print('{0}\n[INFO] Finished calculating high-frequency weekly volatilities.'.
        format('=' * 20 + NOW() + '=' * 20))


if __name__ == '__main__':
    main()