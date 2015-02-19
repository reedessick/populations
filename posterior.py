#!/usr/bin/python
usage = "posterior.py [--options] data.pkl"
description = "computes the posteriors for rates from observed data"
author = "R. Essick (reed.essick@ligo.org)"

#=================================================

import bayes

import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from optparse import OptionParser

#=================================================

parser=OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("", "--maxtau", default=np.infty, type="float")

parser.add_option("", "--test", default=False, action="store_true")

parser.add_option("-g", "--grid", default=False, action="store_true")

parser.add_option("-t", "--tag", default="", type="string")
parser.add_option("-o", "--output-dir", default="./", type="string")

opts, args = parser.parse_args()

#========================

if len(args) != 1:
        raise ValueError("please supply exactly 1 argument")
datafilename = args[0]

it opts.tag:
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
npts = 101
shape = (3,2)
conf = 0.9
eps = 1e-6 ### a small positive definite number to make sure that the covariance matrix is not rank deficient.

### test range functionality

print "range A : conf=%f ,npts=%d"%(conf, npts)
to=time.time()
cih = np.max([posterior.marg_range(datum.dA, confidence=conf)[1] / datum.T for datum in posterior.data ])
cil = max(np.min([posterior.marg_range(max(0, datum.dA-datum.dB), confidence=conf)[0] / datum.T for datum in posterior.data ]), eps)
rA = np.linspace(cil, cih, npts)
#rA = np.linspace(0.06318, 0.10778, npts)
print rA
print "\t", time.time()-to

print "range B : conf=%f ,npts=%d"%(conf, npts)
to=time.time()
cih = np.max([posterior.marg_range(datum.dB, confidence=conf)[1] / datum.T for datum in posterior.data ])
cil = max(np.min([posterior.marg_range(max(0, datum.dB-datum.dA), confidence=conf)[0] / datum.T for datum in posterior.data ]), eps)
rB = np.linspace(cil, cih, npts)
#rB = np.linspace(1e-6, 3.44800000e-02, npts)
print rB
print "\t", time.time()-to

print "range S : conf=%f ,npts=%d"%(conf, npts)
to=time.time()
cih = np.max([posterior.marg_range(min(datum.dA, datum.dB), confidence=conf)[1] / datum.T for datum in posterior.data ])
cil = 0
rS = np.linspace(cil, cih, npts)
#rS = np.linspace(1e-6, 3.44800000e-02, npts)
#rS = np.linspace(1e-6, 0.02, npts)
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
	#ones = np.arange(npts)+1
	to=time.time()
	print posterior(rateA*ones, rateB*ones, rateS*ones)
	print "\t", time.time()-to


	### test marginalization

	print "marg_rateA"
	to=time.time()
	print posterior.marg_rateA(rateB, rateS, rateA=rA)
	print "\t", time.time()-to

	print "marg_rateB"
	to=time.time()
	print posterior.marg_rateB(rateA, rateS, rateB=rB)
	print "\t", time.time()-to

	print "marg_rateS"
	to=time.time()
	print posterior.marg_rateS(rateA, rateB, rateS=rS)
	print "\t", time.time()-to

	print "marg_rateA_rateB"
	to=time.time()
	print posterior.marg_rateA_rateB(rateS, rateA=rA, rateB=rB)
	print "\t", time.time()-to

	print "marg_rateA_rateS"
	to=time.time()
	print posterior.marg_rateA_rateS(rateB, rateA=rA, rateS=rS)
	print "\t", time.time()-to

	print "marg_rateB_rateS"
	to=time.time()
	print posterior.marg_rateB_rateS(rateA, rateB=rB, rateS=rS)
	print "\t", time.time()-to

#=================================================
if opts.verbose:
	print "plotting posteriors"
ones = np.ones(2)
N = npts/5


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

### rateA
if opts.verbose:
        print "rateA"

to=time.time()

fig = plt.figure()
ax = plt.subplot(1,1,1)

if opts.verbose:
        print "\tcomputing post"
post = np.empty_like(rA)
for i in xrange(len(rA)):
	t0 = time.time()
        post[i] = np.sum([ posterior.marg_rateB_rateS(rA[i], rateS=rS, rateB=rB[i*N:(i+1)*N]) for i in xrange(len(rB)/N+1) ])
	print time.time()-t0

ax.plot(rS, post)

ax.set_xlabel("$\lambda_A$")

ax.grid(opts.grid)

ylim = ax.get_ylim()

ax.plot(rateS*ones, ylim, 'k', alpha=0.5)

figname = "%s/rateA%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t", figname
fig.savefig(figname)
plt.close(fig)

print "\t", time.time()-to

if opts.verbose:
        print "\tcomputing post"
### rateB 
if opts.verbose:
        print "rateB"

to=time.time()

fig = plt.figure()
ax = plt.subplot(1,1,1)

post = np.empty_like(rB)
for i in xrange(len(rB)):
        post[i] = posterior.marg_rateA_rateS(rB[i], rateA=rA, rateS=rS)

ax.plot(rB, post)

ax.set_xlabel("$\lambda_B$")

ax.grid(opts.grid)

ylim = ax.get_ylim()

ax.plot(rateS*ones, ylim, 'k', alpha=0.5)

figname = "%s/rateB%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t", figname
fig.savefig(figname)
plt.close(fig)

print "\t", time.time()-to

### rateS
if opts.verbose:
        print "rateA"

to=time.time()

fig = plt.figure()
ax = plt.subplot(1,1,1)

if opts.verbose:
        print "\tcomputing post"
post = np.empty_like(rS)
for i in xrange(len(rS)):
        post[i] = posterior.marg_rateA_rateB(rS[i], rateA=rA, rateB=rB)

ax.plot(rS, post)

ax.set_xlabel("$\lambda_S$")

ax.grid(opts.grid)

ylim = ax.get_ylim()

ax.plot(rateS*ones, ylim, 'k', alpha=0.5)

figname = "%s/rateS%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t", figname
fig.savefig(figname)
plt.close(fig)

print "\t", time.time()-to

#========================
### 2D posteriors
#========================
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
		post[i,j] = posterior.marg_rateS(rA[i], rB[j], rateS=rS)

ax.contour(RA, RB, post)

ax.set_xlabel("$\lambda_A$")
ax.set_ylabel("$\labmda_B$")

ax.grid(opts.grid)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.plot(xlim, rateB*ones, 'k', alpha=0.5)
ax.plot(rateA*ones, ylim, 'k', alpha=0.5)

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
		post[i,j] = posterior.marg_rateB(rA[i], rS[j], rateB=rB)

ax.contour(RB, RS, post)

ax.set_xlabel("$\lambda_A$")
ax.set_ylabel("$\labmda_S$")

ax.grid(opts.grid)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.plot(xlim, rateS*ones, 'k', alpha=0.5)
ax.plot(rateA*ones, ylim, 'k', alpha=0.5)

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
		post[i,j] = posterior.marg_rateA(rB[i], rS[j], rateA=rA)

ax.contour(RB, RS, post)

ax.set_xlabel("$\lambda_B$")
ax.set_ylabel("$\labmda_S$")

ax.grid(opts.grid)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.plot(xlim, rateS*ones, 'k', alpha=0.5)
ax.plot(rateB*ones, ylim, 'k', alpha=0.5)

figname = "%s/rateB-rateS%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t", figname
fig.savefig(figname)
plt.close(fig)

print "\t", time.time()-to

#=================================================
### else?
raise StandardError("WRITE ME")

"""
plot quantiles?
coverage plots?
biases at low sampling statistics?
else?
"""


