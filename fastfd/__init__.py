'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright (c) 2020-2021 Stefan Meili
MIT License


Expose public objects 


PEP0440 compatible formatted version, see:
https://www.python.org/dev/peps/pep-0440/
'''

__version__ = '0.1'

from ._utils import SparseLib
sparse_lib = SparseLib()

from ._modelmatrix import ModelMatrix
from ._discretizedscalar import DiscretizedScalar
from ._axis import LinearAxis
from ._scalar import Scalar
from ._fdmodel import FDModel

__all__ = ['sparse_lib', 'LinearAxis', 'Scalar', 'FDModel', 'DiscretizedScalar', 'ModelMatrix']
