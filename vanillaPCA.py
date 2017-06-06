import numpy as np
import matplotlib.pyplot as plt

def getFileLocs():
    fileLoc1 = r"C:\Users\Jiacheng Z\Dropbox\Courses\17Spring\MS&E349\MS&E349_Shared\HW2\code\Simulation_Dataset_1.csv"
    fileLoc2 = r"C:\Users\Jiacheng Z\Dropbox\Courses\17Spring\MS&E349\MS&E349_Shared\HW2\code\Simulation_Dataset_2.csv"
    fileLoc3 = r"C:\Users\Jiacheng Z\Dropbox\Courses\17Spring\MS&E349\MS&E349_Shared\HW2\code\Simulation_Dataset_3.csv"
    return [fileLoc1, fileLoc2, fileLoc3]

def main():
    FileLocs = getFileLocs()
    for idx, fileLoc in enumerate(FileLocs):
        dataset = np.genfromtxt(fileLoc, delimiter=',')

        CovarianceMat = np.cov(dataset,rowvar=False)
        EigenValues, EigenVectors = np.linalg.eig(CovarianceMat)

        plt.figure()
        plt.plot(EigenValues)
        plt.title('Fig {}.1 Eigenvalues for Dataset {}'.format(idx+1,idx+1))
        plt.savefig('Fig_{}_1.png'.format(idx+1))
        Loadings = (EigenVectors[:3]).T
        Factors = dataset.dot(Loadings.dot(np.linalg.inv((Loadings.T).dot(Loadings))))
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
        ax1.set_title('Fig {}.2 Factors Dataset {}'.format(idx+1, idx+1))
        plt.savefig('Fig_{}_2.png'.format(idx+1))


if __name__ == '__main__':
    main()