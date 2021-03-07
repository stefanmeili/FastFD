'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright 2021 - Stefan Meili
MIT License
'''



from . import sparse_lib


class ModelMatrix:
    '''
    A container for a sparse coefficient matrix. In most use cases, several of these objects are stacked to build a full
    model coefficient matrix solved by FDModel.solve()
    
    Users generally do not call this object directly, but it is eventually produced from a Scalar object 
    '''
    def __init__(self, matrix):
        self.matrix = matrix
    
    
    #Deal with inverse mathematical ops with numpy arrays
    __array_ufunc__ = None
    
    
    #multiplication
    def __mul__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix * B
            return ModelMatrix(matrix)
        
        elif isinstance(B, sparse_lib.np.ndarray):
            if B.size == self.matrix.shape[0]:
                matrix = self.matrix.multiply(B.reshape(-1,1))
                return ModelMatrix(matrix)
            
            else:
                raise ValueError(f"ndarray with shape {B.shape} is not compatible with ModelMatrix with shape {self.matrix.shape}")
            
        else:
            raise NotImplementedError(f"{type(self)} cannot be multiplied to {type(B)}")
    
    def __rmul__(self, B):
        return self * B
    
    
    #division
    def __truediv__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix / B
            return ModelMatrix(matrix)
        
        elif isinstance(B, sparse_lib.np.ndarray):
            if B.size == self.matrix.shape[0]:
                matrix = self.matrix.multiply(1 / B.reshape(-1,1))
                return ModelMatrix(matrix)
            
            else:
                raise ValueError(f"ndarray with shape {B.shape} is not compatible with ModelMatrix with shape {self.matrix.shape}")
                
                
        else:
            raise NotImplementedError(f"{type(self)} cannot be divided by {type(B)}")
    
    def __rtruediv__(self, B):
        raise NotImplementedError(f"{type(self)} cannot be divided by {type(B)}")
        
    
    #addition
    def __add__(self, B):
        if isinstance(B, ModelMatrix):
            matrix = self.matrix + B.matrix
            return ModelMatrix(matrix)
        else:
            raise NotImplementedError(f"{type(self)} cannot be added to {type(B)}")
    
    
    #subtraction
    def __sub__(self, B):
        if isinstance(B, ModelMatrix):
            matrix = self.matrix - B.matrix
            return ModelMatrix(matrix)
        else:
            raise NotImplementedError(f"{type(self)} cannot be added to {type(B)}")
            
    
    # negation
    def __neg__(self):
        return ModelMatrix(-self.matrix)
