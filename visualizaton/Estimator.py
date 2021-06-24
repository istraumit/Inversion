import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import simps
from scipy.stats import chi2
from bisect import *
from DataCube import *

class Estimator:

    def __init__(self, data_cube):
        self.cube = data_cube
    
    def P_matrix_interp(self, K=5):
        C = self.cube
        c_min = min(C.data.flatten())
        M = C.data - c_min
        
        interpol = RegularGridInterpolator(tuple(C.axes), C.data)
        axes_fine = [np.linspace(min(ax), max(ax), K*len(ax)-(K-1)) for ax in C.axes]
        
        IC = np.zeros(tuple([len(ax) for ax in axes_fine]))
        for i in range(IC.shape[0]):
            for j in range(IC.shape[1]):
                for k in range(IC.shape[2]):
                    IC[i,j,k] = interpol( (axes_fine[0][i], axes_fine[1][j], axes_fine[2][k]) )
        
        dc = DataCube(C.axis_labels)
        dc.axes = axes_fine
        dc.data = np.exp(-IC)
        return dc
        
    def P_matrix(self):
        C = self.cube
        c_min = min(C.data.flatten())
        M = C.data - c_min
        dc = DataCube(C.axis_labels)
        dc.axes = C.axes
        dc.data = np.exp(-0.5*M)
        return dc
        
    def marginalize(self):
        P = self.P_matrix()
        M1 = np.zeros((P.data.shape[0],))
        M2 = np.zeros((P.data.shape[1],))
        M3 = np.zeros((P.data.shape[2],))
        
        for i in range(len(M1)): M1[i] = np.nansum(np.nansum(P.data[i,:,:]))
        for i in range(len(M2)): M2[i] = np.nansum(np.nansum(P.data[:,i,:]))
        for i in range(len(M3)): M3[i] = np.nansum(np.nansum(P.data[:,:,i]))
        
        return (M1, M2, M3)

    def get_CDFs(self):
        interpolate = False
        K = 10 # interpolation parameter
        PDF = self.marginalize()
        CDF = []
        ipdf=0
        for pdf in PDF:
            cdf = np.zeros((len(pdf),))
            axis = self.cube.axes[ipdf]
            for i in range(1, len(pdf)): cdf[i] = np.nansum(pdf[0:i])
            if interpolate:
                INTER = interp1d(axis, cdf, kind='quadratic', assume_sorted=True)
                #INTER = Rbf(axis, cdf, function='inverse')
                axis_new = np.linspace(min(axis), max(axis), K*len(axis)-(K-1))
                cdf = INTER(axis_new)
                axis = axis_new
            
            cdf_max = max(cdf)
            cdf = cdf/cdf_max
            CDF.append( (axis, cdf) )
            ipdf += 1
        return CDF

    def get_percentiles(axis_cdf_tuple, perc):
        axis, cdf = axis_cdf_tuple
        vv = []
        for p in perc:
            p_frac = 0.01*p
            i = bisect_right(cdf, p_frac)
            if i==len(cdf): i -= 1
            f = ( p_frac-cdf[i-1])/(cdf[i]-cdf[i-1] )
            vv.append( axis[i-1] + f*(axis[i]-axis[i-1]) )
        return vv
        
    def get_parameter_estimates(self, fmt, percentile_points=[16, 50, 100-16]):
        pp = percentile_points
        CDFs = self.get_CDFs()
        PAR = {}
        for i in range(len(CDFs)):
            pv = Estimator.get_percentiles(CDFs[i], pp)
            med = pv[1]
            
            high = pv[2]-pv[1]
            low = pv[1]-pv[0]
            fm = fmt[i]+' +'+fmt[i]+' -'+fmt[i]
            PAR[self.cube.axis_labels[i]] = fm%(med, high, low)
        return PAR
