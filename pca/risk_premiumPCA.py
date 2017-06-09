# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 16:47:39 2017

@author: Jiacheng Z
"""
#----this correspond to question 2, part 5----
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as invert
from vanillaPCA import getFileLocs

def main():
    FileLocs = getFileLocs()
    Gamma = 50
    FactorCollection = dict()
    LoadingCollection = dict()

    for idx, fileLoc in enumerate(FileLocs):
        dataset = np.genfromtxt(fileLoc, delimiter=',')
        T,N = dataset.shape

        X_Bar = np.mean(dataset,axis=0)
        CovarianceMat = (dataset.T).dot(dataset)/T+Gamma*X_Bar.dot(X_Bar.T)
        EigenValues, EigenVectors = np.linalg.eig(CovarianceMat)

        plt.figure()
        plt.plot(EigenValues)
        plt.title('Fig {}.1 Eigenvalues for Dataset {}'.format(idx+4,idx+1))
        plt.savefig('Fig_{}_1.png'.format(idx+4))

        Loadings = (EigenVectors[:3]).T
        Factors = dataset.dot(Loadings).dot(invert(Loadings.T.dot(Loadings)))

        FactorsForPlot = Factors.T
        trueFactor = np.genfromtxt(r"C:\Users\Jiacheng Z\Dropbox\Courses\17Spring\MS&E349\MS&E349_Shared\HW2\code\Simulation_Factor_{}.csv".
                                   format(idx+1),delimiter=',').T

        plt.figure()
        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        for idy, axis in enumerate([ax1,ax2, ax3]):
            factor = FactorsForPlot[idy]
            if np.mean(factor) < 0: factor = -factor
            axis.plot(factor, label='Fitted ' + r'$\hat{F}$' + '{}'.format(idy+1))
            axis.plot(trueFactor[idy], label='True ' + r'$F_{}$'.format(idy+1))
            axis.legend()
        ax1.set_title('Fig {}.2 Factors Dataset {}'.format(idx+4, idx+1))
        plt.savefig('Fig_{}_2.png'.format(idx+4))

        CollectionKey = 'dataset' + str(idx+1)
        FactorCollection[CollectionKey] = Factors
        LoadingCollection[CollectionKey] = Loadings
    return FactorCollection, LoadingCollection


if __name__ == '__main__':
    FactorCollection, LoadingCollection = main()