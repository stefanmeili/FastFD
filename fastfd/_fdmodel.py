'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright (c) 2020-2021 Stefan Meili
MIT License
'''



from . import sparse_lib
from . import ModelMatrix
from . import DiscretizedScalar

class FDModel:
    def __init__(self, scalars, timestep = None):
        '''
        Builds and solves a finite difference model.
        
        Inputs:
            scalars = [list of Scalars]
                Sets the size of the model. Scalars assigned to a model before coefficient matrices etc. are generated
                to ensure that they are indexed to the appropriate block of variables in the model matrix.
        '''
        
        self.scalars = {}
        for s in scalars:
            if s.name in self.scalars:
                raise ValueError(f'Scalar {s.name} is duplicated')
            self.scalars.update({s.name:s})
                
        self.shape = tuple(s.size for s in self.scalars.values())
        self.size = sum(self.shape)
        self.timestep = timestep
        
        # Register Model pointer in each Scalar
        for s in self.scalars.values():
            s.model = self
            if self.timestep is not None:
                s.timestep = self.timestep
        
        self.coords = {key:scalar.coords for key, scalar in self.scalars.items()} 
        
        # Equations and boundary conditions specified as a dict. This allows only part of the model to be updated between
        # iterations which dramatically improves execution time.
        self.equations = {}
        self.bocos = {}
        
        self.equation_coefficients_built = False
        self.equation_constraints_built = False
        self.boco_coefficients_applied = False
        self.boco_coefficients_applied = False
        
    
    
    def update_equations(self, equations, purge = False):
        '''
        Update the model coefficient matrix and constraint dict that is later used to build the model.
        
        Inputs:
            equations = {'key': (coefficients, constraints)}
                Update one or more of the equations in the model. Coefficients and constraints can be specified as None
                to re-use the existing values. This minimizes overheads in updating the model between iterations.
                
            purge = bool
                wipes the existing equation dict before iterating over equations. Useful in model building and error
                checking to change order of elements and whatnot.
        '''
        if purge:
            self.equations = {}
            self.equation_coefficients_built = False
            self.equation_constraints_built = False
        
        
        for key, (coeff, const) in equations.items():
            old_coeff, old_const = self.equations.get(key, (None, None))
            
            # Check and assign coefficients
            if isinstance(coeff, ModelMatrix):
                new_coeff = coeff.matrix
                self.equation_coefficients_built = False
            elif isinstance(coeff, DiscretizedScalar):
                new_coeff = coeff.to_model_matrix().matrix
                self.equation_coefficients_built = False
            elif coeff is None:
                new_coeff = old_coeff
            else:
                raise TypeError('Matrix coefficients must be [ModelMatrix, DiscritizedScalar, or None]')
            
            # Check and assign constraints
            if isinstance(const, sparse_lib.np.ndarray):
                new_const = const
                self.equation_constraints_built = False
            elif isinstance(const, float) or isinstance(const, int):
                new_const = sparse_lib.np.ones(new_coeff.shape[0]) * const
                self.equation_constraints_built = False
            elif const is None:
                new_const = old_const
            else:
                raise TypeError('Matrix constraints must be [ndarray, float, int, or None]')
        
            self.equations.update({key: (new_coeff, new_const)})
        
        # if coefficients or constraints have been modified, reset all boundary condition applied flags
        for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
            if not self.equation_coefficients_built:
                coeff_applied = False
            if not self.equation_constraints_built:
                const_applied = False
            self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
    
    
    
    def update_bocos(self, bocos, purge = False):
        '''
        Update the model boundary condition dict that is later used to build the model.
        
        Inputs:
            bocos = {'key': (mask, coefficients, constraints)}
                Update one or more of the boundary conditions applied to the model. Masks, coefficients and constraints
                can be specified as None to re-use the existing values. This minimizes overheads in updating the model
                between iterations.
                
            purge = bool
                wipes the existing equation dict before iterating over bocos. Useful in model building and error checking
                to change order of elements and whatnot.
        '''
        
        if purge:
            self.bocos = {}
            self.boco_coefficients_applied = False
            self.boco_constraints_applied = False
            
        for key, (mask, coeff, const) in bocos.items():
            old_mask, old_coeff, old_const, coeff_applied, const_applied = self.bocos.get(key, (None, None, None, False, False))
            
            # Check and assign coefficients to boco_masks
            if isinstance(mask, ModelMatrix) or isinstance(mask, DiscretizedScalar):
                # Define masks using linear algebra instead of setting by slice - much faster
                boco_mask = mask.matrix.T if isinstance(mask, ModelMatrix) else mask.to_model_matrix().matrix.T
                vec_mask = sparse_lib.np.squeeze(sparse_lib.np.array(boco_mask.sum(axis = 1)))
                coeff_mask = sparse_lib.sparse.diags(1 - vec_mask)
                
                new_mask = (coeff_mask, boco_mask, vec_mask.astype(bool))
                
                self.boco_coefficients_applied = False
                self.boco_constraints_applied = False
                coeff_applied = False
                const_applied = False
            
            elif mask is None:
                new_mask = old_mask
            
            else:
                raise TypeError("Boundary condition mask for '{key}' must be [ModelMatrix, DiscritizedScalar, or None]")
            
            # Check and assign coefficients to boco_coeff
            if isinstance(coeff, ModelMatrix) or isinstance(coeff, DiscretizedScalar):
                
                new_coeff = coeff.matrix if isinstance(coeff, ModelMatrix) else coeff.to_model_matrix().matrix
                self.boco_coefficients_applied = False
                coeff_applied = False
            
            elif coeff is None:
                new_coeff = old_coeff
            
            else:
                raise TypeError("Boundary condition coefficients for '{key}' must be [ModelMatrix, DiscritizedScalar, or None]")
            
            # Check and assign constraints to boco_const
            if isinstance(const, sparse_lib.np.ndarray):
                new_const = const.ravel() if const.ndim > 1 else const
                self.boco_constraints_applied = False
                const_applied = False
            
            elif isinstance(const, float) or isinstance(const, int):
                new_const = sparse_lib.np.ones(new_coeff.shape[0]) * const
                self.boco_constraints_applied = False
                const_applied = False
            
            elif const is None:
                new_const = old_const
            
            else:
                raise TypeError("Boundary condition constraints for '{key}' must be [ndarray, float, int, or None]")
            
            
            self.bocos.update({key: (new_mask, new_coeff, new_const, coeff_applied, const_applied)})
    
    
    def _check_equation(self, key, coeff, const, check_type = 'equation'):
        # simple checks of equation dict entries to reduce code repetition
        if coeff is None:
            raise Exception(f"Coefficient matrix for {check_type} '{key}' has not been specified")
        if const is None:
            raise Exception(f"Constraint vector matrix for {check_type} '{key}' has not been specified")
        if coeff.shape[0] != const.size:
            raise Exception(f"Number of rows in {check_type} '{key}' coefficient matrix must equal constraint vector length. Got shapes: {coeff.shape}, {const.shape}")
    
    
    def _check_boco(self, key, mask, coeff, const):
        # simple checks of boco dict entries to reduce code repetition
        if mask is None:
            raise Exception(f"Mask for boundary condition '{key}' has not been specified")
        self._check_equation(key, coeff, const, check_type = 'boundary condition')
        
        
    
    def build(self):
        '''
        Builds the model coefficient matrix and constraint vector that is then solved.
        This process is done in four parts, with the raw coefficient matrix and constraint vectors cached before boundary
        conditions are applied. This can significantly reduce model update times between iterations.
        '''
        
        # If changes have been made to the equation coefficients, rebuild the coefficient matrix
        if not self.equation_coefficients_built:
            eq_coefficients = []
            for key, (coeff, const) in self.equations.items():
                self._check_equation(key, coeff, const)                
                eq_coefficients.append(coeff)
                
            self._coefficients = sparse_lib.sparse.vstack(eq_coefficients, format = 'csr')
        
            if self._coefficients.shape[0] < self.size:
                raise Exception(f"Solution underspecified. Got {self._coefficients.shape[0]} equations and {self.size} unknowns")
            if self._coefficients.shape[0] > self.size:
                raise Exception(f"Solution overspecified. Got {self._coefficients.shape[0]} equations and {self.size} unknowns")
            
            self.equation_coefficients_built = True
            self.boco_coefficients_applied = False
            
        # If changes have been made to the model constraint vector, rebuild the constraint vector
        if not self.equation_constraints_built:
            eq_constraints = []
            for key, (coeff, const) in self.equations.items():
                self._check_equation(key, coeff, const)       
                eq_constraints.append(const)
                
            self._constraints = sparse_lib.np.hstack(eq_constraints)
            
            self.equation_constraints_built = True
            self.boco_constraints_applied = False
            
        # If changes have been made to the boundary condition coefficients, or if the coefficient matrix has been
        # rebuilt, re-apply the boundary conditions
        if not self.boco_coefficients_applied:
            self.coefficients = self._coefficients.copy()
            for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
                if coeff_applied: continue
                self._check_boco(key, mask, coeff, const)
                
                coeff_mask, boco_mask, vec_mask = mask
                #self.coefficients[vec_mask] = coeff  # SLOOOOW!
                self.coefficients = coeff_mask * self.coefficients + boco_mask * coeff
                
                coeff_applied = True
                self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
        
            self.boco_coefficients_applied = True
        
        # Apply boundary condition constraints
        if not self.boco_constraints_applied:
            self.constraints = self._constraints.copy()
            for key, (mask, coeff, const, coeff_applied, const_applied) in self.bocos.items():
                if const_applied: continue
                self._check_boco(key, mask, coeff, const)
                
                coeff_mask, boco_mask, vec_mask = mask
                self.constraints[vec_mask] = const
                
                const_applied = True
                self.bocos.update({key: (mask, coeff, const, coeff_applied, const_applied)})
                
            self.boco_constraints_applied = True
        
    
    
    def solve(self, solver = 'spsolve'):
        '''
        Build and then solve the model. Return the solution.
        '''
        self.build()
        if solver == 'spsolve':
            soln = sparse_lib.linalg.spsolve(self.coefficients, self.constraints)
        elif solver == 'lsqr':
            soln = sparse_lib.linalg.lsqr(self.coefficients, self.constraints)[0]
            
        
        output = {}
        i = 0
        for key, scalar in self.scalars.items():
            output.update({
                key:
                soln[i:i + scalar.size].reshape(scalar.shape)
            })
            i += scalar.size
            
        return output
