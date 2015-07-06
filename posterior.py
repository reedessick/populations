#!/usr/bin/python
usage = "posterior.py [--options] data.pkl"
description = "computes the posteriors for rates from observed data"
author = "R. Essick (reed.essick@ligo.org)"

#=================================================

import bayes
analytics = bayes.analytics

import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from optparse import OptionParser

#=================================================

parser=OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("", "--maxtau", default=np.infty, type="float")

parser.add_option("", "--test", default=False, action="store_true")

parser.add_option("-g", "--grid", default=False, action="store_true")

parser.add_option("", "--oneD", default=False, action="store_true")
parser.add_option("", "--twoD", default=False, action="store_true")

parser.add_option("-t", "--tag", default="", type="string")
parser.add_option("-o", "--output-dir", default="./", type="string")

parser.add_option("-n", "--npts", default=20, type="int")

parser.add_option("", "--exclude-Nc", default=False, action="store_true")

opts, args = parser.parse_args()

#========================

if len(args) != 1:
        raise ValueError("please supply exactly 1 argument")
datafilename = args[0]

if opts.tag:
	opts.tag = "_%s"%opts.tag

#=================================================
if opts.verbose:
        print "===================================================="
        print "  loading data from %s"%(datafilename)
        print "===================================================="

file_obj = open(datafilename, "r")
params = pickle.load(file_obj)
data = pickle.load(file_obj)
file_obj.close()

Ndata = len(data)

### read off taus
### assumes that we have at least one trial
taus = np.array( sorted([key for key in data[0].keys() if isinstance(key, (int,float)) and key <= opts.maxtau]) )

### compute expected rates
dur = params["dur"]
rateS = params["rateS"]
rateA = params["rateA"]
rateB = params["rateB"]

### number of slides
Rs = dict( (tau, data[0][tau]["slideDur"]/dur) for tau in taus )

#=================================================
if opts.verbose: 
	print "constructing posterior object"

posterior = bayes.Posterior()

for datum in data:
	a = len(datum["A"])
	b = len(datum["B"])
	s = len(datum["S"])
	for tau in taus:
		Nc = datum[tau]["num_C"]
		Np = datum[tau]["num_Cp"]
		Nm = datum[tau]["num_Cm"]
			
		posterior.append( bayes.Datum(a+s, b+s, Nc, Np, Nm, tau, T=dur, R=Rs[tau], p=1.0, q=1.0) )

#=================================================
if opts.verbose:
	print "testing functionality"

import time
shape = (3,2)
conf = 0.9
eps = 1e-6 ### a small positive definite number to make sure that the covariance matrix is not rank deficient.

### test range functionality

print "range A : conf=%f ,npts=%d"%(conf, opts.npts)
to=time.time()
#cih = np.max([posterior.marg_range(datum.dA, confidence=conf)[1] / datum.T for datum in posterior.data ])
#cil = max(np.min([posterior.marg_range(max(0, datum.dA-datum.dB), confidence=conf)[0] / datum.T for datum in posterior.data ]), eps)
#rA = np.linspace(cil, cih, opts.npts)
rA = np.logspace(np.log10(0.0995), np.log10(0.1002), opts.npts)
print rA
print "\t", time.time()-to

print "range B : conf=%f ,npts=%d"%(conf, opts.npts)
to=time.time()
#cih = np.max([posterior.marg_range(datum.dB, confidence=conf)[1] / datum.T for datum in posterior.data ])
#cil = max(np.min([posterior.marg_range(max(0, datum.dB-datum.dA), confidence=conf)[0] / datum.T for datum in posterior.data ]), eps)
#rB = np.linspace(cil, cih, opts.npts)
rB = np.logspace(np.log10(0.0995), np.log10(0.1002), opts.npts)
print rB
print "\t", time.time()-to

print "range S : conf=%f ,npts=%d"%(conf, opts.npts)
to=time.time()
#cih = np.max([posterior.marg_range(min(datum.dA, datum.dB), confidence=conf)[1] / datum.T for datum in posterior.data ])
#cil = 0
#rS = np.linspace(cil, cih, opts.npts)
rS = np.logspace(np.log10(0.00095), np.log10(0.00110), opts.npts)
print rS
print "\t", time.time()-to

if opts.test:

	### test call functionality 

	print "posterior.__call__"
	to=time.time()
	print posterior(rateA, rateB, rateS)
	print "\t", time.time()-to

	print "posterior.__call__(1x%s array)"%str(shape)
	ones = np.ones(shape)
	#ones = np.arange(opts.npts)+1
	to=time.time()
	print posterior(rateA*ones, rateB*ones, rateS*ones)
	print "\t", time.time()-to


	### test marginalization

	print "marg_rateA"
	to=time.time()
	print posterior.marg_rateA(rateB, rateS, rateA=rA, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

	print "marg_rateB"
	to=time.time()
	print posterior.marg_rateB(rateA, rateS, rateB=rB, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

	print "marg_rateS"
	to=time.time()
	print posterior.marg_rateS(rateA, rateB, rateS=rS, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

	print "marg_rateA_rateB"
	to=time.time()
	print posterior.marg_rateA_rateB(rateS, rateA=rA, rateB=rB, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

	print "marg_rateA_rateS"
	to=time.time()
	print posterior.marg_rateA_rateS(rateB, rateA=rA, rateS=rS, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

	print "marg_rateB_rateS"
	to=time.time()
	print posterior.marg_rateB_rateS(rateA, rateB=rB, rateS=rS, exclude_Nc=opts.exclude_Nc)
	print "\t", time.time()-to

#=================================================
if opts.verbose:
	print "plotting posteriors"
ones = np.ones(2)
N = opts.npts/2


### too memory intensive!
'''
### meshgrid ranges to get grid points
RB, RS, RA = np.meshgrid(rB, rS, rA)

if opts.verbose:
	print "computing likelihoods at all points on the grid"
to=time.time()
likelihoods = posterior(RA, RB, RS)
print "\t", time.time()-to
'''



#========================
### 1D posteriors
#========================
if opts.oneD:
	### rateA
	if opts.verbose:
	        print "rateA"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
	        print "\tcomputing post"

	post = np.empty_like(rA)
	for j in xrange(len(rA)):
	        post[j] = posterior.marg_rateB_rateS(rA[j], rateB=rB, rateS=rS, exclude_Nc=opts.exclude_Nc)
#       	 post[j] = np.sum([ posterior.marg_rateB_rateS(rA[i], rateS=rS, rateB=rB[i*N:(i+1)*N], exclude_Nc=opts.exclude_Nc) for i in xrange(len(rB)/N+1) ])

	post -= analytics.sum_logs(post)
	post = np.exp(post)

	ax.plot(rA, post, marker='o')

	ax.set_xlabel("$\lambda_A$")
	ax.set_ylabel("$p(\lambda_A)$")

	ax.grid(opts.grid, which="both")

	ylim = ax.get_ylim()

	ax.plot(rateA*ones, ylim, 'r', alpha=0.5)
	
#	ax.set_xscale('log')

	ax.set_ylim(ylim)
	ax.set_xlim(xmin=np.min(rA), xmax=np.max(rA))

	figname = "%s/rateA%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
        	print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

	### rateB 
	if opts.verbose:
	        print "rateB"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
        	print "\tcomputing post"
	post = np.empty_like(rB)
	for j in xrange(len(rB)):
        	post[j] = posterior.marg_rateA_rateS(rB[j], rateA=rA, rateS=rS, exclude_Nc=opts.exclude_Nc)

	post -= analytics.sum_logs(post)
	post = np.exp(post) 

	ax.plot(rB, post, marker='o')

	ax.set_xlabel("$\lambda_B$")
	ax.set_ylabel("$p(\lambda_B)$")

	ax.grid(opts.grid, which="both")

	ylim = ax.get_ylim()

	ax.plot(rateB*ones, ylim, 'r', alpha=0.5)

#	ax.set_xscale('log')

	ax.set_ylim(ylim)
	ax.set_xlim(xmin=np.min(rB), xmax=np.max(rB))

	figname = "%s/rateB%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
        	print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

	### rateS
	if opts.verbose:
        	print "rateS"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
        	print "\tcomputing post"
	post = np.empty_like(rS)
	for j in xrange(len(rS)):
        	post[j] = posterior.marg_rateA_rateB(rS[j], rateA=rA, rateB=rB, exclude_Nc=opts.exclude_Nc)

	post -= analytics.sum_logs(post)
	post = np.exp(post)

	ax.plot(rS, post, marker='o')

	ax.set_xlabel("$\lambda_S$")
	ax.set_ylabel("$p(\lambda_S)$")

	ax.grid(opts.grid, which="both")

	ylim = ax.get_ylim()

	ax.plot(rateS*ones, ylim, 'r', alpha=0.5)

#	ax.set_xscale('log')

	ax.set_ylim(ylim)
	ax.set_xlim(xmin=np.min(rS), xmax=np.max(rS))
	
	figname = "%s/rateS%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
	        print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

#========================
### 2D posteriors
#========================
if opts.twoD:

	### rateA vs rateB
	if opts.verbose:
		print "rateA vs rateB"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
		print "\tcomputing post"
	RA, RB = np.meshgrid(rA, rB)
	post = np.empty_like(RA)
	for i in xrange(len(rA)):
		for j in xrange(len(rB)):
			post[i,j] = posterior.marg_rateS(rA[i], rB[j], rateS=rS, exclude_Nc=opts.exclude_Nc)

	post -= analytics.sum_logs([analytics.sum_logs(post[i]) for i in xrange(len(rA))])
	post = np.exp(post)

	ax.plot(RA, RB, marker='o', linestyle='none', markeredgecolor='k', markerfacecolor='none')
	ax.contour(RA, RB, post, norm = LogNorm())
#	ax.contour(RA, RB, post)

	ax.set_xlabel("$\lambda_A$")
	ax.set_ylabel("$\lambda_B$")

	ax.grid(opts.grid, which="both")

	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	ax.plot(xlim, rateB*ones, 'k', alpha=0.5)
	ax.plot(rateA*ones, ylim, 'k', alpha=0.5)

#	ax.set_xscale('log')
#	ax.set_yscale('log')

	ax.set_xlim(xmin=np.min(rA), xmax=np.max(rA))
	ax.set_ylim(ymin=np.min(rB), ymax=np.max(rB))

	figname = "%s/rateA-rateB%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
		print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

	### rateA vs rateS
	if opts.verbose:
		print "rateA vs rateS"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
		print "\tcomputing post"
	RA, RS = np.meshgrid(rA, rS)
	post = np.empty_like(RA)
	for i in xrange(len(rA)):
		for j in xrange(len(rS)):
			post[i,j] = posterior.marg_rateB(rA[i], rS[j], rateB=rB, exclude_Nc=opts.exclude_Nc)

	post -= analytics.sum_logs([analytics.sum_logs(post[i]) for i in xrange(len(rA))])
	post = np.exp(post)

	ax.plot(RA, RS, marker='o', linestyle='none', markeredgecolor='k', markerfacecolor='none')
	ax.contour(RA, RS, post, norm=LogNorm())
#	ax.contour(RA, RS, post)

	ax.set_xlabel("$\lambda_A$")
	ax.set_ylabel("$\lambda_S$")

	ax.grid(opts.grid, which="both")
	
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	ax.plot(xlim, rateS*ones, 'k', alpha=0.5)
	ax.plot(rateA*ones, ylim, 'k', alpha=0.5)

#	ax.set_xscale('log')
#	ax.set_yscale('log')

	ax.set_xlim(xmin=np.min(rA), xmax=np.max(rA))
	ax.set_ylim(ymin=np.min(rS), ymax=np.max(rS))

	figname = "%s/rateA-rateS%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
		print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

	### rateB vs rateS
	if opts.verbose:
		print "rateB vs rateS"

	to=time.time()

	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	if opts.verbose:
		print "\tcomputing post"
	RB, RS = np.meshgrid(rB, rS)
	post = np.empty_like(RB)
	for i in xrange(len(rB)):
		for j in xrange(len(rS)):
			post[i,j] = posterior.marg_rateA(rB[i], rS[j], rateA=rA, exclude_Nc=opts.exclude_Nc)

	post -= analytics.sum_logs([analytics.sum_logs(post[i]) for i in xrange(len(rB))])
	post = np.exp(post)

	ax.plot(RB, RS, marker='o', linestyle='none', markeredgecolor='k', markerfacecolor='none')
	ax.contour(RB, RS, post, norm=LogNorm())
#	ax.contour(RB, RS, post)

	ax.set_xlabel("$\lambda_B$")
	ax.set_ylabel("$\lambda_S$")

	ax.grid(opts.grid, which="both")

	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	ax.plot(xlim, rateS*ones, 'k', alpha=0.5)
	ax.plot(rateB*ones, ylim, 'k', alpha=0.5)

#	ax.set_xscale('log')
#	ax.set_yscale('log')

	ax.set_xlim(xmin=np.min(rB), xmax=np.max(rB))
	ax.set_ylim(ymin=np.min(rS), ymax=np.max(rS))

	figname = "%s/rateB-rateS%s.png"%(opts.output_dir, opts.tag)
	if opts.verbose:
		print "\t", figname
	fig.savefig(figname)
	plt.close(fig)

	print "\t", time.time()-to

#=================================================
### else?

"""
plot quantiles?
coverage plots?
biases at low sampling statistics?
else?
"""


