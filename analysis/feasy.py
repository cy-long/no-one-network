"""
Python implementation for the feasibility analysis of ecological interactions
"""

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError

def feasibility_community(inte: np.ndarray):
	if np.linalg.matrix_rank(inte) < inte.shape[0]:
		raise ValueError("The interaction matrix is singular")

	nsp = inte.shape[0]
	mu = ro.FloatVector(np.zeros(nsp))
	cov = np.linalg.inv(inte @ inte.T)
	cov = ro.r.matrix(ro.FloatVector(cov.flatten()), nrow=nsp, ncol=nsp)
	a = ro.FloatVector(np.zeros(nsp))
	b = ro.FloatVector(np.full(nsp, np.inf))

	mvtnorm = importr('mvtnorm')
	
	try:
		Omega = mvtnorm.pmvnorm(lower=a, upper=b, mean=mu, sigma=cov)
		return Omega[0]
	except (RRuntimeError, ValueError) as e:
		print(f"An error occurred during pmvnorm execution: {e}")
		return np.nan


def sample_inte_norm(nsp: int, conne:float = 1, mean:float = 0, sigma:float = 1) -> np.ndarray:
    """
    Generate a random interaction matrix
    nsp: size of the system (shape of the inte matrix)
    conne: connectance of the interaction matrix (default: 1)
    mean: scale of the interaction matrix (default: 0)
    """
    inte = np.random.normal(mean, sigma, (nsp, nsp))
    turnon = np.random.binomial(1, conne, (nsp, nsp))
    np.fill_diagonal(turnon, 1)
    inte = inte * turnon
    np.fill_diagonal(inte, -np.absolute(np.diag(inte)))
    return inte