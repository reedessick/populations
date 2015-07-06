dbescription = """ written to house posteriors and priors for bayesian integrals given data """
author = "R. Essick (ressick@mit.edu)"

#=================================================

import analytics
np = analytics.np

#=================================================

#=================================================
# basic utility functions
#=================================================
def nonzero(x):
	"""
	returns 0 if x <0, 1 otherwise
	"""
	return np.array( x >= 0).astype(float)

#=================================================
# classes
#=================================================
class Datum(object):
	"""
	an object that represents on set of observations
	"""

	###
	def __init__(self, dA, dB, Nc, Np, Nm, tau, T=1.0, R=1.0, p=1.0, q=1.0):
		self.dA = dA
		self.dB = dB
		self.Nc = Nc
		self.Np = Np
		self.Nm = Nm
		self.tau = tau
		self.T = T
		self.R = R
		self.p = p
		self.q = q

###
class Prior(object):
	"""
	an object that represents a prior for (rateA, rateB, rateS)
	"""

	###
	def __init__(self, rateA_func=lambda rateA: np.log(nonzero(rateA)), \
		           rateB_func=lambda rateB: np.log(nonzero(rateB)), \
		           rateS_func=lambda rateS: np.log(nonzero(rateS)), \
		           tau_func  =lambda tau: np.log(nonzero(tau)), \
		           T_func =lambda T: np.log(nonzero(T)), \
		           R_func =lambda R: np.log(nonzero(R))
		    ):
		"""
		the default functions are delegations to nonzero.
		These are technically ``improper priors'', and are not normalized
		"""

		self.rateA = rateA_func
		self.rateB = rateB_func
		self.rateS = rateS_func
		self.tau = tau_func
		self.T = T_func
		self.R = R_func

	###
	def __call__(self, rateA, rateB, rateS, tau, T, R):
		"""
		computes the log prior
		"""
		return self.rateA(rateA) + self.rateB(rateB) + self.rateS(rateS) + self.tau(tau) + self.T(T) + self.R(R)

###
class Posterior(object):
	"""
	an object that represents a posterior for (rateA, rateB, rateS)
	contains methods that will:
		numerically marginalize over all observations to obtain an estimate at any given (rateA, rateB, rateS)
		numerically marginalize over all observations and with marginalize over up to 2 rates by integration over poisson credible regions given observations (dA, dB)
	"""

	###
	def __init__(self, data=[], approx="gaussian", prior=Prior()):
		
		### enusre that data has correct form: a list of data instances
		if not isinstance(data, list):
			data = [data]
		if data and not isinstance(data[0], Datum):
			raise ValueError("data not understood. Must be a list of instances of the \"data\" class")
			
		self.data = data

		if not isinstance(prior, Prior):
			raise ValueError("prior must be an instance of the Prior class")
		self.prior = prior

		if approx not in ["gaussian"]:
			raise ValueError("approx=%s not understood"%approx)
		self.approx = approx

	###
	def append(self, datum):
		"""
		add a new observation to the posterior
		"""
		self.data.append( datum )

	###
	def __len__(self):
		return len(self.data)

	###
	def __call__(self, rateA, rateB, rateS, exclude_Nc=False):
		"""
		computes the log posterior

		WARNING! there may be a faster implementation of this using arrays...
		"""
		if self.approx=="gaussian":
			post = 0.0
			for datum in self.data:
				post +=  analytics.gaussian_likelihood(datum.dA, datum.dB, datum.Nc, datum.Np, datum.Nm, \
			                                      rateA, rateB, rateS, datum.tau, T=datum.T, R=datum.R, \
			                                      p=datum.p, q=datum.q, exclude_Nc=exclude_Nc) \
				         + self.prior(rateA, rateB, rateS, datum.tau, datum.T, datum.R)
			return post
		else:
			raise ValueError("approx=%s not understood"%self.approx)

	###
	def marg_range(self, d, confidence=0.99):
		"""
		finds the range of values for a given rate that correspond to a credible interval
		delegates to analytics.poisson_credible_interval
		"""
		(low, high), cum = analytics.poisson_credible_interval(d, confidence=confidence)
		return low, high

	###
	def marg_rateA(self, rateB, rateS, rateA=None, npts=1001, conf=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for (rateB, rateS)
		"""
		if rateA==None:
			### find the range of the marginalization
			cih = np.max([self.marg_range(datum.dA, confidence=conf)[1] / datum.T for datum in self.data ])
			cil = np.min([self.marg_range(max(0, datum.dA-datum.dB), confidence=conf)[0] / datum.T for datum in self.data ])
			rateA = np.linspace(cil, cih, npts)

		### compute marginalization through direct sum
		return analytics.sum_logs([ self(rA, rateB, rateS, exclude_Nc=exclude_Nc) for rA in rateA ]) 
		
	###		
	def marg_rateB(self, rateA, rateS, rateB=None, npts=1001, conf=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for (rateA, rateS)
		"""
		if rateB==None:
			### find the range of the marginalization
			cih = np.max([self.marg_range(datum.dB, confidence=conf)[1] / datum.T for datum in self.data ])
			cil = np.min([self.marg_range(max(0, datum.dB-datum.dA), confidence=conf)[0] / datum.T for datum in self.data ])
			rateB = np.linspace(cil, cih, npts)
		### compute marginalization through direct sum
		return analytics.sum_logs([ self(rateA, rB, rateS, exclude_Nc=exclude_Nc) for rB in rateB ]) 	

	###
	def marg_rateS(self, rateA, rateB, rateS=None, npts=1001, conf=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for (rateA, rateB)
		"""
		if rateS==None:
			### find the range of the marginalization
			cih = np.max([self.marg_range(min(datum.dA, datum.dB), confidence=conf)[1] / datum.T for datum in self.data ])
			cil = 0
			rateS = np.linspace(cil, cih, npts)

		### compute marginalization through direct sum
		return analytics.sum_logs([ self(rateA, rateB, rS, exclude_Nc=exclude_Nc) for rS in rateS ]) 

	###
	def marg_rateA_rateB(self, rateS, rateA=None, rateB=None, nptsA=1001, nptsB=1001, confA=0.99, confB=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for rateS
		"""
		if rateA==None:
			### find the range of the marginalization
			cihA = np.max([self.marg_range(datum.dA, confidence=confA)[1] / datum.T for datum in self.data])
			cilA = np.min([self.marg_range(max(0, datum.dA-datum.dB), confidence=conf)[0] / datum.T for datum in self.data ])
			rateA = np.linspace(cilA, cihA, nptsA)

		if rateB==None:
			cihB = np.max([self.marg_range(datum.dB, confidence=confB)[1] / datum.T for datum in self.data])
			cilB = np.min([self.marg_range(max(0, datum.dB-datum.dA), confidence=conf)[0] / datum.T for datum in self.data ])
			rateB = np.linspace(cilB, cihB, nptsB)

		rateA, rateB = np.meshgrid(rateA, rateB)
		rateS = rateS*np.ones_like(rateA)

		### compute marginalization through direct sum
		return analytics.sum_logs(self(rateA, rateB, rateS, exclude_Nc=exclude_Nc).flatten())

	###
	def marg_rateA_rateS(self, rateB, rateA=None, rateS=None, nptsA=1001, nptsS=1001, confA=0.99, confS=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for rateB
		"""
		if rateA==None:
			### find the range of the marginalization
			cihA = np.max([self.marg_range(datum.dA, confidence=confA)[1] / datum.T for datum in self.data])
			cilA = np.min([self.marg_range(max(0, datum.dA-datum.dB), confidence=conf)[0] / datum.T for datum in self.data ])
			rateA = np.linspace(cilA, cihA, nptsA)
		if rateS==None:
			cihS = np.max([self.marg_range(min(datum.dA, datum.dB), confidence=confA)[1] / datum.T for datum in self.data])
			cilS = 0
			rateS = np.linspace(cilS, cihS, nptsS)

		rateA, rateS = np.meshgrid(rateA, rateS)
		rateB = rateB*np.ones_like(rateA)

		### compute marginalization through direct sum
		return analytics.sum_logs(self(rateA, rateB, rateS, exclude_Nc=exclude_Nc).flatten())
	
	###
	def marg_rateB_rateS(self, rateA, rateB=None, rateS=None, nptsB=1001, nptsS=1001, confB=0.99, confS=0.99, exclude_Nc=False):
		"""
		compute the marginalized posterior for rateA
		"""
		if rateB==None:
			### find the range of the marginalization
			cihB = np.max([self.marg_range(datum.dB, confidence=confB)[1] / datum.T for datum in self.data])
			cilB = np.min([self.marg_range(max(0, datum.dB-datum.dA), confidence=conf)[0] / datum.T for datum in self.data ])
			rateB = np.linspace(cilB, cihB, nptsB)
		if rateS==None:
			cihS = np.max([self.marg_range(min(datum.dA, datum.dB), confidence=confA)[1] / datum.T for datum in self.data])
			cilS = 0
			rateS = np.linspace(cilS, cihS, nptsS)

		rateB, rateS = np.meshgrid(rateB, rateS)
		rateA = rateA*np.ones_like(rateB)

		### compute marginalization through direct sum
		return analytics.sum_logs(self(rateA, rateB, rateS, exclude_Nc=exclude_Nc).flatten())



