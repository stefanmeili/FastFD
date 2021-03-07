'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright 2021 - Stefan Meili
MIT License
'''



from . import sparse_lib
from . import ModelMatrix

class DiscretizedScalar:
    def __init__(self, matrix, scalar):
        '''
        Contains a discritized representation of a Scalar derivative. This class is generally only used internally by
        FastFD.
        
        Inputs:
            matrix = square sparse matrix (scipy or cupyx, depending on sparse_lib)
            
            scalar = Scalar
        '''
        
        self.matrix = matrix
        self.scalar = scalar
        self.model = scalar.model
        self.shape = scalar.shape
        
    
    def to_model_matrix(self):
        '''
        Pad the scalar matrix with zeros so that the number of columns matches the model size. This matrix can be stacked
        by ModelMatrix to produce a model coefficient matrix.
        '''
        scaler_idx = list(self.model.scalars.keys()).index(self.scalar.name)
        
        A = sparse_lib.sparse.csr_matrix((self.matrix.shape[0], sum(self.model.shape[:scaler_idx])))
        B = sparse_lib.sparse.csr_matrix((self.matrix.shape[0], sum(self.model.shape[scaler_idx + 1:])))
        
        model_matrix = sparse_lib.sparse.hstack([A, self.matrix, B], format = 'csr')
        return ModelMatrix(model_matrix)
        
        
    # slicing
    def __getitem__(self, slices):
        slice_index = sparse_lib.np.zeros(self.shape, dtype = bool)
        slice_index[slices] = True
        
        matrix = self.matrix[slice_index.ravel()]
        return DiscretizedScalar(matrix, self.scalar).to_model_matrix()
    
    
    # Deal with inverse mathematical ops with numpy arrays
    __array_ufunc__ = None
    
    
    # multiplication
    def __mul__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix * B
            return DiscretizedScalar(matrix, self.scalar)
        
        elif isinstance(B, sparse_lib.np.ndarray):
            if B.shape == self.shape or (B.ndim == 1 and B.size == self.matrix.shape[0]):
                matrix = self.matrix.multiply(B.reshape(-1,1))
                return DiscretizedScalar(matrix, self.scalar)
            else:
                raise ValueError(f"ndarray with shape {B.shape} is not compatible with Scalar '{self.scalar.name}' with shape {self.shape}")
        
        elif isinstance(B, DiscretizedScalar):
            if self.scalar == B.scalar:
                matrix = self.matrix * B.matrix
                return DiscretizedScalar(matrix, self.scalar)
            else:
                raise NotImplementedError(f"DiscretizedScalars can only multiplied if they share the same Scalars. Got: '{self.scalar.name}' and '{B.scalar.name}'")
                
        else:
            raise NotImplementedError(f"{type(self)} '{self.scalar.name}' cannot be multiplied by {type(B)}")
        
    
    def __rmul__(self, B):
        return self * B
    
    
    # division
    def __truediv__(self, B):
        if isinstance(B, int) or isinstance(B, float):
            matrix = self.matrix / B
            return DiscretizedScalar(matrix, self.scalar)
        
        elif isinstance(B, sparse_lib.np.ndarray):
            if B.shape == self.shape or (B.ndim == 1 and B.size == self.matrix.shape[0]):
                matrix = self.matrix.multiply(1 / B.reshape(-1,1))
                return DiscretizedScalar(matrix, self.scalar)
            else:
                raise ValueError(f"ndarray with shape {B.shape} is not compatible with Scalar '{self.scalar.name}' with shape {self.shape}")
        
        else:
            # inverse of a discritized scaler matrix is singular. Can't see use case.
            raise NotImplementedError(f"{type(self)} '{self.scalar.name}' cannot be divided by {type(B)}")
    
    
    def __rtruediv__(self, B):
        raise NotImplementedError(f"{type(B)} cannot be divided by {type(self)} '{self.scalar.name}'")
    
    # addition
    def __add__(self, B):
        if isinstance(B, DiscretizedScalar):
            if self.scalar == B.scalar:
                matrix = self.matrix + B.matrix
                return DiscretizedScalar(matrix, self.scalar)
            else:
                return self.to_model_matrix() + B.to_model_matrix()
            
        elif isinstance(B, ModelMatrix):
            return self.to_model_matrix() + B
        
        else:
            raise NotImplementedError(f"{type(self)} '{self.scalar.name}' cannot be added to {type(B)}")
    
    
    def __radd__(self, B):
        if isinstance(B, ModelMatrix):
            return B + self.to_model_matrix()
        
        else:
            raise NotImplementedError(f"{type(B)} cannot be added to {type(self)} '{self.scalar.name}'")
        
    
    # subtraction
    def __sub__(self, B):
        if isinstance(B, DiscretizedScalar):
            if self.scalar == B.scalar:
                matrix = self.matrix + B.matrix
                return DiscretizedScalar(matrix, self.scalar)
            else:
                return self.to_model_matrix() - B.to_model_matrix()
            
        elif isinstance(B, ModelMatrix):
            return self.to_model_matrix() - B
        
        else:
            raise NotImplementedError(f"{type(B)} cannot be subtracted from {type(self)} '{self.scalar.name}'")
    
    
    def __rsub__(self, B):
        if isinstance(B, ModelMatrix):
            return B - self.to_model_matrix()
        
        else:
            raise NotImplementedError(f"{type(B)} cannot be added to {type(self)} '{self.scalar.name}'")
            
    
    # negation
    def __neg__(self):
        return DiscretizedScalar(-self.matrix, self.scalar)
