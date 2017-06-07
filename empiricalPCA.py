# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 02:37:05 2017

@author: Jiacheng Z
"""
#----this is a wrapper for code in question 3----
import numpy as np
import pandas as pd

Datasets = dict()
Datasets['accrual']=r'https://raw.githubusercontent.com/jiacheng0409/pca/master/accrual.csv'
Datasets['book2market']=r'https://raw.githubusercontent.com/jiacheng0409/pca/master/book2market.csv'
Datasets['industry']=r'https://raw.githubusercontent.com/jiacheng0409/pca/master/industry.csv'

for Key in Datasets.keys():
    rwData = pd.read_csv(Datasets[Key])
    X_Mat = rwData.as_matrix()
    Covariance = np.cov(X_Mat,rowvar=False)
    EigenValues, EigenVectors = np.linalg.eig(Covariance)