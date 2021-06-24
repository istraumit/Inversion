import numpy as np



class DataCube:

    def __init__(self, axis_labels, DEBUG = False):
        assert(len(axis_labels)==3)
        self.axis_labels = axis_labels
        self.DEBUG = DEBUG
        
    def load_from_array(self, A):
        assert(A.shape[1]==len(self.axis_labels)+1)
        Nax = len(self.axis_labels)
        axes = [set() for i in range(Nax)]
        for i in range(Nax):
            for j in range(A.shape[0]):
                axes[i].add( A[j,i] )
        axes = [list(s) for s in axes]
        for ax in axes: ax.sort()
        if self.DEBUG:
            print('Axes:')
            for i in range(len(axes)):
                print('\t', self.axis_labels[i], ':', min(axes[i]), '...', max(axes[i]), 'len=', len(axes[i]))
        D = np.empty( tuple([len(ax) for ax in axes]) )
        D[:,:,:] = np.nan
        if self.DEBUG:
            print('Cube dimensions:', D.shape)
        for i in range(A.shape[0]):
            indices = []
            for j in range(len(axes)):
                ind = axes[j].index(A[i,j])
                indices.append(ind)
            D[tuple(indices)] = A[i,-1]
        self.data = D
        self.axes = axes
        




    













