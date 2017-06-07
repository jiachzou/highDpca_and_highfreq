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
    AAPL = rwData.iloc[:, 1].as_matrix()
    DeltaSPY = np.diff(SPY, n=1)
    DeltaAAPL = np.diff(AAPL, n=1)
    Sq_DeltaSPY = np.square(DeltaSPY)
    Sq_DeltaAAPL = np.square(DeltaAAPL)
    VolLenWeekly = len(SPY)//N_WeeklyObs +1

    WeeklySpotVol = dict()
    WeeklySpotVol['SPY'] = np.array([0.0] * VolLenWeekly)
    WeeklySpotVol['AAPL'] = np.array([0.0] * VolLenWeekly)

    for idx in range(VolLenWeekly):
        if idx+N_WeeklyObs>len(SPY):
            WeeklyChunkSPY = Sq_DeltaSPY[idx*N_WeeklyObs:]
            WeeklyChunkAAPL = Sq_DeltaAAPL[idx*N_WeeklyObs:]
        else:
            WeeklyChunkSPY = Sq_DeltaSPY[idx*N_WeeklyObs:(idx + 1)*N_WeeklyObs]
            WeeklyChunkAAPL = Sq_DeltaAAPL[idx*N_WeeklyObs:(idx + 1)*N_WeeklyObs]

        WeeklySpotVol['SPY'][idx] = np.sum(WeeklyChunkSPY)
        WeeklySpotVol['AAPL'][idx] = np.sum(WeeklyChunkAAPL)

    plt.figure()
    plt.plot(WeeklySpotVol['AAPL'],label='AAPL High-frequency Volatility')
    plt.plot(WeeklySpotVol['SPY'], label='SPY High-frequency Volatility')
    plt.title('Fig.7 Weeekly Volatility')
    plt.ylabel('Quadratic Covariance')
    plt.xlabel('No. of Week')
    plt.legend()
    plt.savefig('Fig_7.png')

    print('{0}\n[INFO] Finished calculating high-frequency weekly volatilities.'.
        format('=' * 20 + NOW() + '=' * 20))

    #------------part 2, qeustion 4: jump estimation-----------------
    N_DailyObs = N_WeeklyObs/5
    VolLenDaily = len(SPY)//N_DailyObs+1
    DailySpotVol = dict()
    DailySpotVol['AAPL'] = np.array([0.0] * VolLenDaily)
    DailySpotVol['SPY'] = np.array([0.0] * VolLenDaily)
    for idx in range(VolLenDaily):
        if idx+N_DailyObs>len(SPY):
            DailyChunkSPY = Sq_DeltaSPY[idx*N_DailyObs:]
            DailyChunkAAPL = Sq_DeltaAAPL[idx*N_DailyObs:]
        else:
            DailyChunkSPY = Sq_DeltaSPY[idx*N_DailyObs:(idx + 1)*N_DailyObs]
            DailyChunkAAPL = Sq_DeltaAAPL[idx*N_DailyObs:(idx + 1)*N_DailyObs]

        DailySpotVol['SPY'][idx] = np.sum(DailyChunkSPY)
        DailySpotVol['AAPL'][idx] = np.sum(DailyChunkAAPL)

    plt.figure()
    plt.plot(DailySpotVol['AAPL'],label='AAPL High-frequency Volatility')
    plt.plot(DailySpotVol['SPY'], label='SPY High-frequency Volatility')
    plt.title('Fig.8 Daily Spot Volatility')
    plt.ylabel('Quadratic Covariance')
    plt.xlabel('No. of Day')
    plt.legend()
    plt.savefig('Fig_8.png')

    print('{0}\n[INFO] Finished calculating high-frequency daily volatilities.'.
        format('=' * 20 + NOW() + '=' * 20))


if __name__ == '__main__':
    main()