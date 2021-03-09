'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright (c) 2020-2021 Stefan Meili
MIT License
'''


from . import sparse_lib


class LinearAxis:
    '''
    Defines one axis that a Scalar is discretized over.
    
    Inputs:
        name = 'string'
            Name of the dimension. Axis names assigned to a Scalar must be unique. However, Axis names can be reused
            so long as the Axes are assigned to different Scalars in the same FDModel.
 
    '''
    # TODO - Accept a 1D numpy (or cupy) array and allow non-uniform grids?
    # TODO - Allow polar coordinates
    
    def __init__(self, name, **kwargs):
        self.name = name
        
        if all([arg in kwargs for arg in ['start', 'stop', 'num']]):
            start, stop, num = kwargs['start'], kwargs['stop'], kwargs['num']
            self.coords = sparse_lib.np.linspace(start, stop, num)
            self.delta = abs(stop - start) / (num - 1)
            
        else:
            raise Exception('Axis range must be supplied as a tuple (start, stop, num)')
            
        
        self.num_points = len(self.coords)
