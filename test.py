#!/usr/bin/python
usage = """run a simulation of poisson processes and coincidence filtering."""
author = "Reed Essick (ressick@mit.edu)"

#=================================================

import series_utils

import numpy as np

from optparse import OptionParser

#=================================================

parser=OptionParser(usage=usage)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("", "--dur", default=1e4, type="float")

parser.add_option("", "--rateS", default=1e-5, type="float")

parser.add_option("", "--rateA", default=0.1, type="float")
parser.add_option("", "--rateB", default=0.1, type="float")

parser.add_option("", "--tau", default=0.01, type="float")

parser.add_option("-n", "--num-trials", default=1, type="int")
parser.add_option("-t", "--time", default=False, action="store_true")

opts, args = parser.parse_args()

if opts.time:
	import time
	t_o = time.time()

#=================================================
n_coincs = []
for n in xrange(opts.num_trials):
	if opts.time:
		to = time.time()

	### simulate series
	if opts.verbose:
		print "simulating series S with rate=%.5f"%(opts.rateS)
	seriesS = series_utils.poisson_series(opts.rateS, opts.dur, tau=opts.tau, flag=True)
	if opts.verbose:
		print "\texpected %.5f\n\tobserved %d"%(opts.rateS*opts.dur, len(seriesS))

	if opts.verbose:
		print "simulating series A with rate=%.5f"%(opts.rateA)
	seriesA = series_utils.poisson_series(opts.rateA, opts.dur, tau=opts.tau, flag=False)
	if opts.verbose:
		print "\texpected %.5f\n\tobserved %d"%(opts.rateA*opts.dur, len(seriesA))

	if opts.verbose:
		print "simulating series B with rate=%.5f"%(opts.rateB)
	seriesB = series_utils.poisson_series(opts.rateB, opts.dur, tau=opts.tau, flag=False)
	if opts.verbose:
		print "\texpected %.5f\n\tobserved %d"%(opts.rateB*opts.dur, len(seriesB))

	### add seriesS to both series A and series B
	if opts.verbose:
		print "coherently adding series S to both series A and series B and clustering"
	seriesAS = series_utils.cluster(seriesA+seriesS, opts.tau)
	seriesBS = series_utils.cluster(seriesB+seriesS, opts.tau)

	#========================
	### perform coinc between series A and series B
	if opts.verbose:
		print "performing coincindence filtering between series A and series B"

	coincs = series_utils.coinc(seriesA, seriesB, opts.tau)

	#=======================
	if opts.verbose:
		print "%d / %d :\n\tfound %d\n\texpected %.5f"%(n+1, opts.num_trials, len(coincs), (opts.rateS + 2*opts.tau*opts.rateA*opts.rateB)*opts.dur)
		if opts.time:
			print "\t", time.time()-to, "sec"

	"""
	HIT IT LIKE A FREQUENTIST:

	want to simulate cyclic time-slides (assume slides by 2*opts.tau)

	for many different realizations of noise:

		for several different tau:
			for many different slides:
				compute n_coinc+
				compute n_coinc-

			average over slides to compute estimate for:
				rate+
				rate-

		fit for (rateA*rateB, rateA+rateB, rateS) simultaneously from rate+, rate- vs. tau

	compute (mean, var) of these statistics over different noise realizations.
		-> get an estimate of the precision with which we can measure such things.

	consider how to accumulate evidence over many different thresholds (like a distribution test).
	"""







	n_coincs.append( len(coincs) )

if opts.time:
	print "simulted %d series in %.3f sec"%(opts.num_trials, time.time()-t_o)

print "exp\t",  (opts.rateS + 2*opts.tau*opts.rateA*opts.rateB)*opts.dur
print "mean\t", np.mean(n_coincs)
print "var\t",  np.var(n_coincs)
