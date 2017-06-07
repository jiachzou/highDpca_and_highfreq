# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 17:09:02 2017

@author: Jiacheng Z
"""
#----this is a wrapper for code in question 2----
import factorSimulation
import vanillaPCA
import risk_premiumPCA
def main(reSimulate=True):
    if reSimulate:  factorSimulation.main()
    vanillaFactors, vanillaLoadings = vanillaPCA.main()
    rpFactors, rpLoadings = risk_premiumPCA.main()
    print('Simulation Done!')

if __name__ == '__main__':
    main()


