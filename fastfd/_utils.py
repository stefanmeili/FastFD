'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright (c) 2020-2021 Stefan Meili
MIT License
'''

class SparseLib:
    '''
    A pointer object that allows this library to switch between scipy and cupy sparse matrix libraries
    
    Possible future support for Dask, TensorFlow, and Pytorch.
    '''
    
    def __init__(self):
        self._np = None
        self._sparse = None
        self._linalg = None
        
        self.initialized = False
        
        
    def __call__(self, sparse_lib):
        global np
        global sparse
        global linalg
        
        if sparse_lib == 'scipy':
            import numpy as np
            import scipy.sparse as sparse
            import scipy.sparse.linalg as linalg
            
        elif sparse_lib == 'cupy':
            import cupy as np
            import cupyx.scipy.sparse as sparse
            import cupyx.scipy.sparse.linalg as linalg
            
        else:
            raise ValueError(f"Sparse library must be one of: ['scipy', 'cupy']")
        
        
        self._np = np
        self._sparse = sparse
        self._linalg = linalg
        
        self.initialized = True
    
    
    def _check_init(self):
        if not self.initialized:
            raise Exception("FastFD has not been initialized. Call 'fastfd.sparse_lib('scipy')' or 'fastfd.sparse_lib('cupy')'")
    
    @property
    def np(self):
        self._check_init()
        return self._np
        
    
    @property
    def sparse(self):
        self._check_init()
        return self._sparse
        
        
    @property
    def linalg(self):
        self._check_init()
        return self._linalg
        
