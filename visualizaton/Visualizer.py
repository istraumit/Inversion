import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, interp2d
from scipy.integrate import simps
from bisect import *

from Estimator import *

class Visualizer:

    def __init__(self, estimator):
        self.est = estimator
        
    def argmax(A):
        S = A.shape
        v_max = -np.inf
        i_max = None
        for i in range(S[0]):
            for j in range(S[1]):
                for k in range(S[2]):
                    v = A[i,j,k]
                    if v>v_max:
                        v_max = v
                        i_max = (i,j,k)
        return i_max

    def argmin(A):
        S = A.shape
        v_min = np.inf
        i_min = None
        for i in range(S[0]):
            for j in range(S[1]):
                for k in range(S[2]):
                    v = A[i,j,k]
                    if v<v_min:
                        v_min = v
                        i_min = (i,j,k)
        return i_min

    def interpol_slice(A, ax1, ax2, K):
        INTER = interpolate.interp2d(ax1, ax2, A.T, kind='cubic')
        
        ax1_new = np.linspace(min(ax1), max(ax1), K*len(ax1)-(K-1))
        ax2_new = np.linspace(min(ax2), max(ax2), K*len(ax2)-(K-1))
        A_inter = INTER(ax1_new, ax2_new)
        
        return A_inter, ax1_new, ax2_new
    
        
    def plot_P_slices(self):
    
        def get_extent(ax1, ax2):
            return [ax1[0], ax1[-1], ax2[0], ax2[-1]]
            
        P = self.est.P_matrix()
        i_max = Visualizer.argmax(P.data)
        #i_max = Visualizer.argmin(P.data)

        i,j,k = i_max
        S1 = P.data[i,:,:]
        S2 = P.data[:,j,:]
        S3 = P.data[:,:,k]
        
        ax1 = P.axes[0]
        ax2 = P.axes[1]
        ax3 = P.axes[2]

        if False:
            K = 1
            ax1_new = np.linspace(min(ax1), max(ax1), K*len(ax1)-(K-1))
            ax2_new = np.linspace(min(ax2), max(ax2), K*len(ax2)-(K-1))
            ax3_new = np.linspace(min(ax3), max(ax3), K*len(ax3)-(K-1))
            
            INTER1 = interp2d(ax2, ax3, S1.T, kind='cubic')
            INTER2 = interp2d(ax1, ax3, S2.T, kind='cubic')
            INTER3 = interp2d(ax1, ax2, S3.T, kind='cubic')
        
            S1_inter = INTER1(ax2_new, ax3_new)
            S2_inter = INTER2(ax1_new, ax3_new)
            S3_inter = INTER3(ax1_new, ax2_new)
        
            S1 = S1_inter; S2 = S2_inter; S3 = S3_inter
            ax1 = ax1_new; ax2 = ax2_new; ax3 = ax3_new
            #S1[S1<0]=0
            #S2[S2<0]=0
            #S3[S3<0]=0
        
        plt.subplot(221)

        bar_lbl = 'Likelihood'
        
        img_inter = 'none'
        im = plt.imshow(S1.T, aspect='auto',  interpolation=img_inter, origin='lower', extent=get_extent(ax2, ax3))
        bar = plt.colorbar(im, aspect=20, shrink=1)
        bar.set_label(bar_lbl)
        plt.locator_params(nbins=5)
        plt.xlabel(P.axis_labels[1])
        plt.ylabel(P.axis_labels[2])
    
        plt.subplot(222)
        
        im = plt.imshow(S2.T, aspect='auto',  interpolation=img_inter, origin='lower', extent=get_extent(ax1, ax3))
        bar = plt.colorbar(im, aspect=20, shrink=1)
        bar.set_label(bar_lbl)
        plt.locator_params(nbins=5)
        plt.xlabel(P.axis_labels[0])
        plt.ylabel(P.axis_labels[2])
    
        plt.subplot(223)
        
        im = plt.imshow(S3.T, aspect='auto',  interpolation=img_inter, origin='lower', extent=get_extent(ax1, ax2))
        bar = plt.colorbar(im, aspect=20, shrink=1)
        bar.set_label(bar_lbl)
        plt.locator_params(nbins=5)
        plt.xlabel(P.axis_labels[0])
        plt.ylabel(P.axis_labels[1])
    
        #plt.show()

    def plot_PDFs(self):
        PDF = self.est.marginalize()
        for i in range(len(PDF)):
            plt.subplot(221+i)
            plt.plot(self.est.cube.axes[i], PDF[i])
            plt.xlabel(self.est.cube.axis_labels[i])
            
        plt.show()
        
    def plot_CDFs(self):
        perc_points = [16, 50, 100-16]
        CDF = self.est.get_CDFs()
        
        for i in range(len(CDF)):
            perc_values = Estimator.get_percentiles(CDF[i], perc_points)
        
            plt.subplot(221+i)
            plt.plot(CDF[i][0], CDF[i][1], linewidth=2)
            
            for y in perc_points: plt.axhline(0.01*y, linestyle='--')
            for x in perc_values: plt.axvline(x, linestyle='--')
            
            plt.xlabel(self.est.cube.axis_labels[i])
            

