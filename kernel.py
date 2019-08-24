import numpy as np
import numpy.linalg as la


class Kernel:
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def _polykernel(dimension, offset):
        return lambda x, y: (offset + np.inner(x, y)) ** dimension

    @classmethod
    def inhomogenous_polynomial(cls, dimension):
        return cls._polykernel(dimension=dimension, offset=1.0)

    @classmethod
    def homogenous_polynomial(cls, dimension):
        return cls._polykernel(dimension=dimension, offset=0.0)

    @classmethod
    def linear(cls, x1, x2):
        return np.inner(x1, x2)

    @classmethod
    def gaussian(cls, sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        return lambda x, y: np.tanh(kappa * np.dot(x, y) + c)

    @staticmethod
    def radial_basis(gamma=10):
        return lambda x, y: np.exp(-gamma*la.norm(np.subtract(x, y)))
