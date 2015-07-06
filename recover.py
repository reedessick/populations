#!/usr/bin/python
usage = """recover.py [--options] data.pkl"""
description = """written to recover populations of events from poisson series"""
author = "R. Essick"

import os
import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from optparse import OptionParser

#=================================================

figwidth = 15
figheight = 8

axpos = [0.15, 0.15, 0.8, 0.8]

axpos1 = [0.15, 0.15, 0.35, 0.8]
axpos2 = [0.50, 0.15, 0.35, 0.8]

#=================================================
parser = OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-g", "--grid", default=False, action="store_true")

parser.add_option("", "--max-tau", default=np.infty, type="float")

parser.add_option("-t", "--tag", default="", type="string")
parser.add_option("-o", "--output-dir", default="./", type="string")

opts, args = parser.parse_args()

#=================================================

if len(args) != 1:
        raise ValueError("please supply exactly 1 argument")
datafilename = args[0]

if opts.tag:
        opts.tag = "_%s"%opts.tag

if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

#=================================================
if opts.verbose:
        print "===================================================="
        print "  loading data from %s"%(datafilename)
        print "===================================================="

file_obj = open(datafilename, "r")
params = pickle.load(file_obj)
data = pickle.load(file_obj)
file_obj.close()

### read off taus
### assumes that we have at least one trial
taus = np.array( sorted([key for key in data[0].keys() if isinstance(key, (int,float)) if key <= opts.max_tau]) )

Ndata = len(data)
Ntaus = len(taus)

### compute expected rates
dur = params["dur"]
rateS = params["rateS"]
rateA = params["rateA"]
rateB = params["rateB"]
rateC = {}
rateCp = {}
rateCm = {}
for tau in taus:
        rateC[tau] = rateS + 2*tau*rateA*rateB
        rateCp[tau] = 2*tau*(rateA+rateS)*(rateB+rateS)
        rate_accident = 2*tau*rateA*rateB
        rateCm[tau] = 2*tau*(rateA-rate_accident)*(rateB-rate_accident)

#=================================================
if opts.verbose:
	print "===================================================="
        print "  fitting each noise instantiation"
        print "===================================================="

### observed rates
orateC = np.array([[datum[tau]["num_C"]/dur for tau in taus] for datum in data])
orateCp = np.array([[datum[tau]["num_Cp"]/datum[tau]["slideDur"] for tau in taus] for datum in data])
orateCm = np.array([[datum[tau]["num_Cm"]/datum[tau]["slideDur"] for tau in taus] for datum in data])

### fit for Cm
"""
taus2 = np.sum( taus**2 )
taus3 = np.sum( taus**3 )
taus4 = np.sum( taus**4 )
det = taus2*taus4 - taus3**2

ratetaus = np.sum(orateCm * taus, axis=1 )
ratetaus2 = np.sum( orateCm * taus**2, axis=1 )

a = (taus4* ratetaus - taus3*ratetaus2)/det
b = (-taus3*ratetaus + taus2*ratetaus2)/det

### check chi2 for goodness of fit
chi2 = np.sum( (orateCm - np.outer(a, taus) - np.outer(a, taus**2))**2/(np.outer(a, taus) + np.outer(a, taus**2)), axis=1)
if np.any(chi2 > 0.01):
	raise StandardError("chi2 is too big for some fits!")
"""
### fit a third order polynomial to relieve pressure on the quadratic term?
### DOES NOT WORK WELL and if anything hurts the measurement of the quadratic term.
_, Cm_b, Cm_a = np.polyfit( taus, np.transpose(orateCm/taus, (1,0)), 2) ### no biases, but big variances
#Cm_b, Cm_a = np.polyfit( taus, np.transpose(orateCm/taus, (1,0)), 1) ### introduces a bias in the quadratic term

### fit for Cp
#Cp_a = np.polyfit( taus, np.transpose(orateCp/taus, (1,0)), 0) ### linear with no offset
Cp_a = 1.0*np.sum(orateCp/taus, axis=1)/Ntaus ### faster computation?

### fit for C
C_a, C_o = np.polyfit( taus, np.transpose(orateC, (1,0)), 1) ### linear with offset

#=================================================
if opts.verbose:
	print "plotting fit parameters"

### histogram a, b values
fig = plt.figure(figsize=(figwidth,figheight))
axa = fig.add_axes(axpos1)
axb = fig.add_axes(axpos2)

figS = plt.figure(figsize=(figwidth,figheight))
ax = figS.add_axes(axpos)

figCp = plt.figure(figsize=(figwidth,figheight))
axCp = figCp.add_axes(axpos)

figC = plt.figure(figsize=(figwidth, figheight))
axC1 = figC.add_axes(axpos1)
axC2 = figC.add_axes(axpos2)

nbins = max(Ndata/10, 1)

### plot
axa.hist( 1 - Cm_a/(2*rateA*rateB), nbins, histtype="step" )
axb.hist( 1 - Cm_b/(-4*rateA*rateB*(rateA+rateB)), nbins, histtype="step" )

ax.plot( 1 - Cm_a/(2*rateA*rateB), 1 - Cm_b/(-4*rateA*rateB*(rateA+rateB)), marker="o", markerfacecolor="none", linestyle="none")

axCp.hist( 1 - Cp_a/(2*(rateA+rateS)*(rateB+rateS)), nbins, histtype="step")
ylim = axCp.get_ylim()
x = 1 - rateA*rateB/((rateA+rateS)*(rateB+rateS))
axCp.plot( x*np.ones(2), ylim, "k--")
axCp.text(x, ylim[1], "$1-\\frac{\lambda_A\lambda_B}{\left(\lambda_A+\lambda_S\\right)\left(\lambda_B + \lambda_S\\right)}$", ha="left", va="top")

axC1.hist( 1 - C_a/(2*(rateA*rateB + rateS*rateB + rateA*rateS)), nbins, histtype="step")
axC2.hist( 1 - C_o/rateS, nbins, histtype="step")

### label
axa.set_ylabel("count")
axa.set_xlabel("$1 - \\frac{a}{2\lambda_A\lambda_B}$")

axb.set_ylabel("count")
axb.set_xlabel("$1 - \\frac{b}{-4\lambda_A\lambda_B\left(\lambda_A+\lambda_B\\right)}$")
axb.yaxis.tick_right()
axb.yaxis.set_label_position("right")

ax.set_ylabel("$1 - \\frac{b}{-4\lambda_A\lambda_B\left(\lambda_A+\lambda_B\\right)}$")
ax.set_xlabel("$1 - \\frac{a}{2\lambda_A\lambda_B}$")

axCp.set_ylabel("count")
axCp.set_xlabel("$1 - \\frac{a}{2\left(\lambda_A+\lambda_S\\right)\left(\lambda_B + \lambda_S\\right)}$")

axC1.set_ylabel("count")
axC1.set_xlabel("$1 - \\frac{a}{2\left(\lambda_A\lambda_B+\lambda_A\lambda_S+\lambda_S\lambda_B\\right)}$")

axC2.set_ylabel("count")
axC2.set_xlabel("$1 - \\frac{o}{\lambda_S}$")
axC2.yaxis.tick_right()
axC2.yaxis.set_label_position("right")

### decorate
axa.grid(opts.grid)
axb.grid(opts.grid)

ax.grid(opts.grid)

axCp.grid(opts.grid)

axC1.grid(opts.grid)
axC2.grid(opts.grid)

### save
figname = "%s/Cm-fit_params-hist%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t", figname
fig.savefig(figname)
plt.close(fig)

figname = "%s/Cm-fit_params-scatter%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t", figname
figS.savefig(figname)
plt.close(figS)

figname = "%s/Cp-fit_params-hist%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t", figname
figCp.savefig(figname)
plt.close(figCp)

figname = "%s/C-fit_params-hist%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t", figname
figC.savefig(figname)
plt.close(figC)

#=================================================
"""
perform a hypothesis test on C+ data using fit parameters -> p-values
	fit C+ data to extract "(rateA+rateS)*(rateB+rateS)"
	perform null test with C- data as distribution. Need to include fitting uncertainty from C+ params
perform a hypothesis test on C data using fit parameters -> p-values
	fit C data to extract "rateS" and "rateA*rateB"
	perform null test on "rateS" to see whether we can detect a signal this way
	perform null test on rateA*rateB to using C- data as distribution. Need to include fitting uncertainty from C params

Can we write a "quick" MCMC for (rateA, rateB, rateS) that attempts to fit the data and recover parameters?
	need distributions of errors for each point along the "X vs. tau" curves.
		gaussian?
		errors are correlated between different points...
Instead, just use a "joint chi2" minimization using all data.
	still need relative variances on data points to weight the chi2 appropriately.

"""
#=================================================
"""
Compare the sensitivity of these different methods as a function of rateA, rateB, dur, number of slides, etc.
	quote the fraction of trials for which we detected a signal.
	rinse and repeat with various parametric combinations -> sigmoid detection curve
	rinse and repeat with various dur -> require observing time to detect a given population
"""
