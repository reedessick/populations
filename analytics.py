usage="""module to store known moments from our analytical model, as well as other useful functions"""

#=================================================

import numpy as np

from math import factorial

#=================================================
#
# basic utility functions
#
#=================================================

def sum_logs(vec, axis=-1):
	if not isinstance(vec, np.ndarray):
		vec = np.array(vec)
	max_vec = np.max(vec)
	return np.log(np.sum(np.exp(vec-max_vec), axis=axis)) + max_vec

def fast_fact(x, xmax=50):
        """
        computes the factorial of x using sterling's approximation if x > xmax
        """
        if x > xmax: ### use sterling's approximation
                return (2*np.pi*x)**0.5 * (x/np.exp(1))**x
        else:
                return factorial(x)

def log_fast_fact(x, xmax=50):
	if x > xmax:
		return 0.5*np.log(2*np.pi) + (x+0.5)*np.log(x) - x
	else:
		return np.log(factorial(x))

def poisson(x, mean, xmax=1000):
        """
        computes the poisson probability of observing x given mean.
        if x > xmax, we use a gaussian approximation

        returns the log of the probability
        """
        if x > xmax:
                return -0.5*np.log(2*np.pi*mean) - 0.5*(x-mean)**2/mean
        else:
                return x*np.log(mean) - log_fast_fact(x) - mean

def poisson_credible_interval(d, confidence=0.95):
        """
        finds the credible interval corresponding to "confidence" for the mean a poisson series given the observed number of events defined by "d"
        """
        if confidence < 0 or confidence >= 1:
                raise ValueError("confidence must be between 0 and 1")

        high = np.ceil(d)
        low = np.floor(d)

        phigh = poisson(d, high)
        plow = poisson(d, low)

        cum = 0.0
        while cum < confidence:
                if phigh > plow: ### include high next
                        cum += np.exp(phigh)
                        high += 1
                        phigh = poisson(d, high)
                else: ### include low next
                        cum += np.exp(plow)
                        low -= 1
                        if low < 0:
                                plow = -np.infty
                        else:
                                plow = poisson(d, low)

        ### return credible interval and associated cumulative probability
        return (low+1, high-1), cum


#=================================================
#
# moments for counts
#
#=================================================

#========================
# first moments
#========================

def d(rate, T=1.0):
	return T*rate

def dA(rateA, rateS, T=1.0):
	return d(rateA+rateS, T=T)

def dB(rateB, rateS, T=1.0):
	return dA(rateB, rateS, T=T)

def Nc(rateA, rateB, rateS, tau, T=1.0, p=1.0, q=1.0):
	return T * q * rateS + 2*tau*T * p * (rateA + rateS)*(rateB + rateS)

def Np(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p * (rateA+rateS)*(rateB+rateS)

def Nm(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p * rateA*rateB * np.exp( -2*tau*(rateA+rateB+2*rateS) )

#========================
# variances
#========================

def Vaa(rateA, rateS, T=1.0):
	return dA(rateA, rateS, T=T)

def Vbb(rateB, rateS, T=1.0):
	return Vaa(rateB, rateS, T=T)

def Vcc(rateA, rateB, rateS, tau, T=1.0, p=1.0, q=1.0):
#	if q!=1.0 or q!=1.0:
#		raise ValueError("likelihood thresholds are not yet implemented!")
#	return T*rateS + 2*tau*T*(rateA*rateB + 3*rateS*(rateA+rateB+2*rateS)) \
#		+ 4*tau**2*T*(rateA+rateB+4*rateS)*(rateA+rateS)*(rateB+rateS)
	return T * q**2 * rateS + 4*tau*T * p*q * rateS*(rateA+rateB+2*rateS) \
		+ 2*tau*T * p**2 * (rateS**2 + (rateA+rateS)*(rateB+rateS) ) \
		+ 4*tau**2*T * p**2 * (rateA**2*(rateB+rateS) + rateB**2*(rateA+rateS) + 6*rateA*rateB*rateS \
		                        + 5*rateS**2*(rateA+rateB) + 4*rateS**3 )

def Vpp(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p**2 * (rateS**2 + (rateA+rateS)*(rateB+rateS) \
	                   + 2*tau*R*(rateA+rateS)*(rateB+rateS)*(rateA+rateB+4*rateS) )

def Vmm(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p**2 * rateA*rateB * np.exp(-2*tau*(rateA+rateB+2*rateS)) \
		*(1 + 2*tau*(rateA*(1-np.exp(-2*tau*(rateB+rateS))) + rateB*(1-np.exp(-2*tau*(rateA+rateS))) ) \
		  + 4*tau**2*rateA*rateB*( (1-np.exp(-2*tau*(rateB+rateS)))*(1-np.exp(-2*tau*(rateA+rateS))) \
		                            + np.exp(-2*tau*(rateA+rateB+2*rateS)) ) \
		  + 2*tau*R*( rateB*np.exp(-2*tau*(rateA+rateS)) + rateA*np.exp(-2*tau*(rateB*rateS)) \
		             + 2*tau*rateA*rateB*(np.exp(-2*tau*(rateA+rateS)) + np.exp(-2*tau*(rateB+rateS)) \
		                                    - 4*np.exp(-2*tau*(rateA+rateB+2*rateS)) )  ) ) \

#========================
# covariances
#========================	

def Vab(rateS, T=1.0):
	return T*rateS

def Vac(rateA, rateB, rateS, tau, T=1.0, p=1.0, q=1.0):
	return T * q * rateS + 2*tau*T * p * (rateA+rateS)*(rateB+2*rateS)

def Vbc(rateA, rateB, rateS, tau, T=1.0, p=1.0, q=1.0):
	return Vac(rateB, rateA, rateS, tau, T=T, p=p, q=q)

def Vap(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p * (rateB+2*rateS)*(rateA+rateS)

def Vbp(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return Vap(rateB, rateA, rateS, tau, T=T, R=R, p=p)

def Vam(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p * rateA*rateB*np.exp(-2*tau*(rateA+rateB+2*rateS))*(1 - 2*tau*(rateA+2*rateS))

def Vbm(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return Vam(rateB, rateA, rateS, tau, T=T, R=R, p=p)

def Vcp(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0, q=1.0):
#	if p!=1.0 or q!=1.0:
#		raise ValueError("likelihood thresholds are not yet implemented!")
#	return 2*tau*T*R*( rateS*(rateA+rateB+2*rateS) \
#	                   + 2*tau*(rateA*rateB + rateS*(rateA+rateB+rateS))*(rateA+rateB+4*rateS))
	return 2*tau*T*R * p * ( q * rateS*(rateA+rateB+2*rateS) \
		                  + 2*tau * p *(rateA+rateS)*(rateB+rateS)*(rateA+rateB+4*rateS) )

def Vcm(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0, q=1.0):
#	if p!=1.0 or q!=1.0:
#		raise ValueError("likelihood thresholds are not yet implemented!")
#	return - 8*tau**2*T*R*rateA*rateB*np.exp(-2*tau*(rateA+rateB+rateS)) * (rateS + 2*tau*(rateA*rateB + rateS*(rateA+rateB+rateS)))
	return - 8*tau**2*T*R * p * rateA*rateB*np.exp(-2*tau*(rateA+rateB+2*rateS)) * (q*rateS + 2*tau* p * (rateA + rateS)*(rateB + rateS))


def Vpm(rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0):
	return 2*tau*T*R * p**2 * rateA*rateB*np.exp(-2*tau*(rateA+rateB+2*rateS)) \
		 * ( 1 - 2*tau*rateS + 4*tau**2*(rateA*rateB + rateS*(rateA+rateB+2*rateS)) \
		     + 2*tau*R*(rateA+rateB+2*rateS + 2*tau*(rateS*(rateA+rateB+2*rateS) \
		                                              - 2*(rateA+rateS)*(rateB+rateS))) )

#=================================================
#
# Likelihoods
#
#=================================================
def gaussian_likelihood(d_A, d_B, N_c, N_p, N_m, rateA, rateB, rateS, tau, T=1.0, R=1.0, p=1.0, q=1.0):
	"""
	computes the likelihood through a gaussian approximation. This is known to be in error for small number statistics.
	returns the log likelihood
	"""
	single = isinstance(rateA, (int, float))
	if single:
		rateA = np.array([rateA])
		rateB = np.array([rateB])
		rateS = np.array([rateS])
		
	npts = list(np.shape(rateA))
	N = len(npts)

	### ensure data are arrays
	data = np.reshape( np.outer( np.array( [d_A, d_B, N_c, N_p, N_m] ), np.ones(npts) ), tuple([5]+npts) )

	### compute means
	m_dA = dA(rateA, rateS, T=T)
	m_dB = dB(rateB, rateS, T=T)
	m_Nc = Nc(rateA, rateB, rateS, tau, T=T, p=p, q=q)
	m_Np = Np(rateA, rateB, rateS, tau, T=T, R=R, p=p)
	m_Nm = Nm(rateA, rateB, rateS, tau, T=T, R=R, p=p)

	means = np.array( [ m_dA, m_dB, m_Nc, m_Np, m_Nm ] )

	### compute covariance matrix
	vaa = Vaa(rateA, rateS, T=T)
	vbb = Vbb(rateB, rateS, T=T)
	vcc = Vcc(rateA, rateB, rateS, tau, T=T, p=p, q=q)
	vpp = Vpp(rateA, rateB, rateS, tau, T=T, R=R, p=p)
	vmm = Vmm(rateA, rateB, rateS, tau, T=T, R=R, p=p)

	vab = Vab(rateS, T=T)
	vac = Vac(rateA, rateB, rateS, tau, T=T, p=p, q=q)
	vap = Vap(rateA, rateB, rateS, tau, T=T, R=R, p=p)
	vam = Vam(rateA, rateB, rateS, tau, T=T, R=R, p=p)

	vbc = Vbc(rateA, rateB, rateS, tau, T=T, p=p, q=q)
	vbp = Vbp(rateA, rateB, rateS, tau, T=T, R=R, p=p)
	vbm = Vbm(rateA, rateB, rateS, tau, T=T, R=R, p=p)

	vcp = Vcp(rateA, rateB, rateS, tau, T=T, R=R, p=p, q=q)
	vcm = Vcm(rateA, rateB, rateS, tau, T=T, R=R, p=p)
	
	vpm = Vpm(rateA, rateB, rateS, tau, T=T, R=R, p=p)

	cov = np.array( [ [vaa, vab, vac, vap, vam],
	                  [vab, vbb, vbc, vbp, vbm],
	                  [vac, vbc, vcc, vcp, vcm],
	                  [vap, vbp, vcp, vpp, vpm],
	                  [vam, vbm, vcm, vpm, vmm]
	                ] 
	              )

	cov = np.transpose(cov, tuple(range(2,N+2)+[0,1])) ### just in case we have arrays

	if np.any(np.linalg.det(cov) <= 0): ### we get nan from log(det(cov)) if det(cov) <= 0 (which is unphysical)
		print "rateA=", rateA
		print "rateB=", rateB
		print "rateS=", rateS
		print "tau  =", tau
		print "T    =", T

		print "determinants of sub-matricies are as follows"
		S = "a b c p m".split()
		for ind, s in enumerate(S):
			print s
			cov = np.array(eval("[v%s%s]"%(s,s)))
			print "\t", np.linalg.det(cov)

			for ind1, s1 in enumerate(S[ind+1:]):
				print s, s1
				cov = np.array(eval("[[v%s%s, v%s%s],[v%s%s, v%s%s]]"%(s,s, s,s1, s,s1, s1,s1))).transpose((2,0,1))[0]
				print "\t", np.linalg.det(cov)

				for ind2, s2 in enumerate(S[ind+1+ind1+1:]):
					print s, s1, s2
					cov = np.array( eval("[[v%s%s, v%s%s, v%s%s],[v%s%s, v%s%s, v%s%s],[v%s%s, v%s%s, v%s%s]]"\
				                        %(s,s, s,s1, s,s2, s,s1, s1,s1, s1,s2, s,s2, s1,s2, s2,s2) ) ).transpose((2,0,1))[0]
					print "\t", np.linalg.det(cov)

					for ind3, s3 in enumerate(S[ind+1+ind1+1+ind2+1:]):
						print s, s1, s2, s3
						cov = np.array(eval("[[v%s%s, v%s%s, v%s%s, v%s%s],[v%s%s, v%s%s, v%s%s, v%s%s],[v%s%s, v%s%s, v%s%s, v%s%s], [v%s%s, v%s%s, v%s%s, v%s%s]]"\
                                                        %(s,s, s,s1, s,s2, s,s3, s,s1, s1,s1, s1,s2, s1,s3, s,s2, s1,s2, s2,s2, s2,s3, s,s3, s1,s3, s2,s3, s3,s3))).transpose((2,0,1))[0]
						print "\t", np.linalg.det(cov)

		print " ".join(S)
		cov = np.array([[vaa, vab, vac, vap, vam], [vab, vbb, vbc, vbp, vbm], [vac, vbc, vcc, vcp, vcm], [vap, vbp, vcp, vpp, vpm], [vam, vbm, vcm, vpm, vmm]]).transpose((2,0,1))[0]
		print "\t", np.linalg.det(cov)

		print "eigenvalues"
		for v in np.linalg.eigvals(cov):
			print "\t", v

		raise ValueError("unphysical covariance matrix...")

	### compute gaussian approximation
	### normalization may be wrong!
	G = np.linalg.inv(cov)

	### fancy manipulations to make x look like G
	x = np.transpose(data-means)

	if len(npts) == 1:
		x = np.diagonal( np.reshape( np.outer( x, x ), tuple(npts + [5] + npts + [5]) ) , axis1=0, axis2=2 ).transpose((2,0,1))
	else:
		x = np.reshape( np.outer( x, x ), tuple(npts + [5] + npts + [5]) )
		for i in xrange(N): ### iteratively take the diagonal... 
			x = np.diagonal( x, axis1=0, axis2=N+1-i )
		x = np.transpose( x, tuple(range(2,N+2) + [0, 1]) ) # transpose

	ans = -2.5*np.log(2*np.pi) + 0.5*np.log( np.linalg.det(cov) ) - 0.5*np.sum(np.sum( x * G , axis=-1), axis=-1)
#	ans -2.5*np.log(2*np.pi) + 0.5*np.log( np.linalg.det(cov) ) - 0.5*np.sum( x * np.sum( G * x, axis=1), axis=0)

	if single:
		return ans[0]
	else:
		return ans


def my_det(x):
	if isinstance(x, (int, float)):
		return x
	if len(x) == 1:
		return x[0]

	ans = 0.0

	N = len(x)
	for ind in xrange(N):
		v = x[0, ind]
		minor = np.array( [ [ x[i][j] for j in xrange(N) if j != ind ] for i in xrange(1, N) ] )
		ans += v * my_det(minor) * (-1)**ind

	return ans		



