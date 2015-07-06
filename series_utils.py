usage = """a module to rapidly simulate poisson series and to perform coincidence filtering"""
author = "Reed Essick (ressick@mit.edu)"

#=================================================

import numpy as np
from math import factorial

#=================================================
#
# general utilities 
#
#=================================================
def sloppy_factorial(n, max_n=100):
	"""
	uses sterling approximation if n > max_n
	"""
	if n > max_n: ### sterling
		return (2*np.pi*n)**0.5 * ( n / np.exp(0) )**n
	else: ### full factorial
		return factorial(n)

#=================================================
#
# drawing series and time-shifts
#
#=================================================
def poisson_series(rate, duration, tau=0.0, flag=False):
	"""
	simulates a poisson series with "rate" through the time-span "duration"
	returns a list of times

	we assume a minimum spacing of "tau" between events
		should be reasonably safe for rate*tau << 1
	"""
	if rate*tau > 1e-3:
		if "yes" != raw_input("rate*tau > 1e-3. Exclusion may affect results. Do you want to continue? [yes/no]"):
			raise ValueError("rate*tau > 1e-3")

	if rate < 0:
		raise ValueError("rate must be positive semi-definite")
	elif rate == 0:
		return []
	else:
		series = []
		t = 0.0
		while True:
			t += tau + -np.log( np.random.rand() )/rate ### increment time to next event
			if t < duration: ### keep and move on
				series.append( (t, flag) )
			else: ### out of bounds, break
				break
		return series

###
def cyclic_shift(series, dt, duration, resort=False):
	"""
	perform a cyclic shift of time series data. No livetime will be lost
	if resort, assumes it is already sorted
	"""
	if resort:
		ans = []
		wer = []
		for tup in series:
			t, stuff = tup
			t += dt
			if t < duration:
				wer.append( (t, stuff) )
			else:
				ans.append( (t%duration, stuff) )
		return ans+wer

#		series = [ tuple([(tup[0]+dt)%duration]+list(tup[1:])) for tup in series ]
#		series.sort(key=lambda l: l[0])
#		return series
	else:
		return [ tuple([(tup[0]+dt)%duration]+list(tup[1:])) for tup in series ]

###
def linear_shift(series, dt, duration=False):
	"""
	perform a linear shift of time series data. Livetime will be lost.
	if duration:
		throw away events with times larger than duration
	"""
	return [ tuple([(tup[0]+dt)]+list(tup[1:])) for tup in series if (not duration) or (tup[0]+dt > duration) ]

#=================================================
#
# coincs and clustering
#
#=================================================
def all_coinc(series1, series2, tau, return_list=False):
	"""
	performs brute force coinc, and returns the number of coincs
	assumes lists are orderd chronologically (early to late)
	"""

	N1 = len(series1)
	N2 = len(series2)

	if N1 == 0 or N2 == 0:
		if return_list:
			return []
		else:
			return 0

	### index that tracks with series1
	ind1 = 0
	tup1 = series1[ind1]
	t1 = tup1[0]

	### two indicies that track with series2
	### one that lags the time in series1
	### one that leads the time in series2
	### 
	ind2E = 0
	ind2L = 0
	tup2E = series2[ind2E]
	t2E = tup2E[0]
	tup2L = series2[ind2L]
	t2L = tup2L[0]

	### counter for coincs
	if return_list:
		coincs = []
	else:
		n_coinc = 0

	while True: ### there be dragons here...
		if t2E < t1-tau: ### check whether the earlier index is too early
			ind2E += 1
			if ind2E >= N2: ### we're out of bounds on ind2E, so we break
				break
			tup2E = series2[ind2E]
			t2E = tup2E[0]
			if ind2L < ind2E:
				ind2L = ind2E
				tup2L = series2[ind2L]
				t2L = tup2L[0]
			continue
		if t2L < t1+tau: ### check whether the late index is too late
			if ind2L < N2-1: ### it's ok to increment
				if series2[ind2L+1][0] < t1+tau: ### increment because the next one is still in bounds
					ind2L += 1
					tup2L = series2[ind2L]
					t2L = tup2L[0]
					continue

		### at this point, t2E is early enough and t2L is late enough that they bracket t1
		### we now check that there is still a good coinc
		if t2E <= t1+tau: ### there is at least one good coinc
			### count new coincs
			if return_list:
				coincs += [ (tup1, tup2) for tup2 in series2[ind2E:ind2L+1] ]
			else:
				n_coinc += ind2L - ind2E + 1
#			print "\n"
#			print t2E, t1, tau
#			print "t1-t2E = %.5f"%(t1-t2E)
#			print "t2L-t1 = %.5f"%(t2L-t1)
#			print "ind1, ind2E, ind2L : %d, %d, %d"%(ind1, ind2E, ind2L)
#			raw_input("")

		### increment ind1
		ind1 += 1
		if ind1 < N1:
			tup1 = series1[ind1]
			t1 = tup1[0]
		else:
			break

	if return_list:
		return coincs
	else:
		return n_coinc

###
def single_coinc(series1, series2, tau):
	"""
	returns a list of coincident events, defined as living within the time window tau

	this may not be super efficient...

	assumes no two events from the same series are closer than "tau"
	"""
	coincs = []

	### set up arrays
	N1 = len(series1)
	N2 = len(series2)

	n1 = 0
	n2 = 0
	tup1 = series1[0]
	tup2 = series2[0]
	t1 = tup1[0]
	t2 = tup2[0]
	while (n1 < N1) and (n2 < N2):
		diff = t1 - t2

		if diff > tau: ### make sure we're within bounds
			n2 += 1
			if n2 < N2:
				tup2 = series2[n2]
				t2 = tup2[0]

		elif diff < -tau:
			n1 += 1
			if n1 < N1:
				tup1 = series1[n1]
				t1 = tup1[0]

		else: #abs(diff) <= tau

			if diff == 0: ### perfect match. we'll get no better
				coincs.append( (tup1, tup2) )
				n1 += 1
				if n1 < N1:
					tup1 = series1[n1]
					t1 = tup1[0]
				n2 += 1
				if n2 < N2:
					tup2 = series2[n2]
					t2 = tup2[0]

			elif diff > 0: # t1 > t2
				_n2 = n2 + 1
				if (_n2 < N2) and (abs(series2[_n2][0] - t1) < diff): ### new point is better
					coincs.append( (tup1, series2[_n2]) )
					n1 += 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]
					n2 = _n2+1
					if n2 < N2:
						tup2 = series2[n2]
						t2 = tup2[0]
				else:
					coincs.append( (tup1, tup2) )
					n1 += 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]
					n2 += 1
					if n2 < N2:
						tup2 = series2[n2]
						t2 = tup2[0]
			else: # t2 < t1
				_n1 = n1 + 1
				if (_n1 < N1) and (abs(series1[_n1][0] - t2) < -diff): ### new point is better
					coincs.append( (series1[_n1], tup2) )
					n1 = _n1 + 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]
					n2 += 1
					if n2 < N2:
						tup2 = series2[n2]
						t2 = tup2[0]
				else:
					coincs.append( (tup1, tup2) )
					n1 += 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]
					n2 += 1
					if n2 < N2:
						tup2 = series2[n2]
						t2 = tup2[0]


	'''
			### we only have to look forward in time
			if diff >= 0: # t1 >= t2
				if (n2+1 < N2) and abs(t1-series2[n2+1][0]) < diff:
					n2 += 1
					tup2 = series2[n2]
					t2 = tup2[0]
				else:
					coincs.append( (tup1, tup2) )

					n1 += 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]

					n2 += 1
					if n2 < N2:
						tup2 = series2[0]
						t2 = tup2[0]

			elif diff < 0: # t2 > t1
				if (n1+1 < N1) and abs(t2-series1[n1+1][0]) < diff:
					n1 += 1
					tup1 = series1[n1]
					t1 = tup1[0]
				else:
					coincs.append( (tup1, tup2) )

					n1 += 1
					if n1 < N1:
						tup1 = series1[n1]
						t1 = tup1[0]

					n2 += 1
					if n2 < N2:
						tup2 = series2[0]
						t2 = tup2[0]
	'''
	return coincs

###
def cluster(series, tau):
	"""
	cluster events to there are no two events closer in time than tau
	"""
	### generate separate clusters
	clusters = []
	cluster = []
	for tup in series:
		t = tup[0]
		if cluster:
			if np.any(np.abs(t-np.array([l[0] for l in cluster]))<tau):
				cluster.append( tup )
			else:
				clusters.append( cluster )
				cluster = [ tup ]
		else:
			cluster.append( tup )

	clusters.append( cluster )

	### return the average events
	return [ (np.mean([tup[0] for tup in cluster]), cluster) for cluster in clusters ]

