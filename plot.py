#!/usr/bin/python
usage = """plots.py [--options] data.pkl"""
description = """ulot/analyze the results from main.py """
author = "R. Essick"

import os
import pickle
import numpy as np

import analytics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from optparse import OptionParser

#=================================================

figwidth = 15
figheight = 8

axpos = [0.15, 0.15, 0.7, 0.8]

axpos1 = [0.15, 0.15, 0.35, 0.8]
axpos2 = [0.50, 0.15, 0.35, 0.8]

#=================================================

parser=OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-g", "--grid", default=False, action="store_true")

parser.add_option("", "--individual-tau", default=False, action="store_true")
parser.add_option("", "--max-tau", default=np.infty, type="float")

parser.add_option("-t", "--tag", default="", type="string")
parser.add_option("-o", "--output-dir", default="./", type="string")

parser.add_option("", "--continuity-correction", default=False, action="store_true", help="apply a continuity correction in Gaussian approximation for scatter plots.")

opts, args = parser.parse_args()

#========================

if len(args) != 1:
	raise ValueError("please supply exactly 1 argument")
datafilename = args[0]

if opts.continuity_correction:
	opts.tag = "cc_%s"%opts.tag

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

Ndata = len(data)

### read off taus
### assumes that we have at least one trial
taus = np.array( sorted([key for key in data[0].keys() if isinstance(key, (int,float)) and key <= opts.max_tau]) )

### compute expected rates
dur = params["dur"]
rateS = params["rateS"]
rateA = params["rateA"]
rateB = params["rateB"]
#rateC = {}
#rateCp = {}
#rateCm = {}
#for tau in taus:
#	rateC[tau] = rateS + 2*tau*(rateA*rateB + rateS*(rateA+rateB))
#	rateCp[tau] = 2*tau*(rateA+rateS)*(rateB+rateS)
#	rateCm[tau] = 2*tau*rateA*rateB*(1 - (1-np.exp(-2*tau*rateA)) - (1-np.exp(-2*tau*rateS)))*(1 - (1-np.exp(-2*tau*rateB)) - (1-np.exp(-2*tau*rateS)))

### number of slides
Rs = dict( (tau, data[0][tau]["slideDur"]/dur) for tau in taus )

#=================================================
### plot stuff
if opts.verbose:
	print "===================================================="
	print "  generating plots"
	print "===================================================="

#=======================
# basic sanity plots
#=======================
if opts.verbose: 
	print "\tbasic sanity plots"

figS = plt.figure(figsize=(figwidth,figheight))
ax1S = figS.add_axes(axpos1)
ax2S = figS.add_axes(axpos2)

figA = plt.figure(figsize=(figwidth,figheight))
ax1A = figA.add_axes(axpos1)
ax2A = figA.add_axes(axpos2)

figB = plt.figure(figsize=(figwidth,figheight))
ax1B = figB.add_axes(axpos1)
ax2B = figB.add_axes(axpos2)

nbins = max(Ndata/10, 1)

### plot
ax1S.hist([len(datum["S"]) for datum in data], nbins, histtype="step")
ax2S.hist([(len(datum["S"]) - analytics.d(rateS, dur))/analytics.d(rateS, dur)**0.5 for datum in data], nbins, histtype="step")

ax1A.hist([len(datum["A"]) for datum in data], nbins, histtype="step")
ax2A.hist([(len(datum["A"]) - analytics.d(rateA, dur))/analytics.d(rateA, dur)**0.5 for datum in data], nbins, histtype="step")

ax1B.hist([len(datum["B"]) for datum in data], nbins, histtype="step")
ax2B.hist([(len(datum["B"]) - analytics.d(rateB, dur))/analytics.d(rateB, dur)**0.5 for datum in data], nbins, histtype="step")

### label
ax1S.set_xlabel("No. S")
ax2S.set_xlabel("$z_S$")

ax1A.set_xlabel("No. A")
ax2A.set_xlabel("$z_A$")

ax1B.set_xlabel("No. B")
ax2B.set_xlabel("$z_B$")

ax1S.set_ylabel("count")
ax2S.set_ylabel("count")
ax2S.yaxis.tick_right()
ax2S.yaxis.set_label_position("right")

ax1A.set_ylabel("count")
ax2A.set_ylabel("count")
ax2A.yaxis.tick_right()
ax2A.yaxis.set_label_position("right")

ax1B.set_ylabel("count")
ax2B.set_ylabel("count")
ax2B.yaxis.tick_right()
ax2B.yaxis.set_label_position("right")

### decorate
ax1S.grid(opts.grid)
ax2S.grid(opts.grid)

ax1A.grid(opts.grid)
ax2A.grid(opts.grid)

ax1B.grid(opts.grid)
ax2B.grid(opts.grid)

### save
figname = "%s/S%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figS.savefig(figname)
plt.close(figS)

figname = "%s/A%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figA.savefig(figname)
plt.close(figA)

figname = "%s/B%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figB.savefig(figname)
plt.close(figB)

#========================
# counts as a function of tau
#========================
if opts.verbose:
	print "\tcounts as a function of tau"

figC = plt.figure() ### rate of zero-lag coincs
axC = figC.add_axes(axpos)

figCp = plt.figure() ### rate of slide-coincs with zero-lag
axCp = figCp.add_axes(axpos)

figCm = plt.figure() ### rate of slide-coincs wiout zero-lag
axCm = figCm.add_axes(axpos)

### plot observed data
for datum in data:
	axC.plot( taus, [datum[tau]["num_C"]/dur for tau in taus], '.-')
	axCp.plot( taus, [datum[tau]["num_Cp"]/datum[tau]["slideDur"] for tau in taus], '.-' )
	axCm.plot( taus, [datum[tau]["num_Cm"]/datum[tau]["slideDur"] for tau in taus], '.-' )

### plot expected values as dashed line
fine_taus = np.linspace(taus[0], taus[-1], 1001)

#axC.plot( fine_taus, rateS + 2*fine_taus*rateA*rateB, color='k', linestyle=":")
#axC.plot( fine_taus, rateS + 2*fine_taus*(rateA*rateB + rateS*rateB + rateA*rateS), color='k', linestyle="--")
axC.plot( fine_taus, analytics.Nc(rateA, rateB, rateS, fine_taus), color='k', linestyle="--")

#axCp.plot( fine_taus, 2*fine_taus*(rateA+rateS)*(rateB+rateS), color='k', linestyle="--")
axCp.plot( fine_taus, analytics.Np(rateA, rateB, rateS, fine_taus), color='k', linestyle="--")

#rate_accident = 2*fine_taus*rateA*rateB
#axCm.plot( fine_taus, 2*fine_taus*(rateA-rate_accident)*(rateB-rate_accident), color='k', linestyle="--")
axCm.plot( fine_taus, analytics.Nm(rateA, rateB, rateS, fine_taus), color='k', linestyle="--")

### label
axC.set_xlabel("$\\tau$")
axC.set_ylabel("rate of zero-lag coincidences")

axCp.set_xlabel("$\\tau$")
axCp.set_ylabel("rate of coincs in slides with zero-lag coincs")

axCm.set_xlabel("$\\tau$")
axCm.set_ylabel("rate of coincs in slides without zero-lag coincs")

### decorate
axC.grid(opts.grid)
axCp.grid(opts.grid)
axCm.grid(opts.grid)

### save
figname = "%s/C-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figC.savefig(figname)
plt.close(figC)

figname = "%s/Cp-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figCp.savefig(figname)
plt.close(figCp)

figname = "%s/Cm-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "\t\t", figname
figCm.savefig(figname)
plt.close(figCm)

#========================
# averages
#========================
if opts.verbose:
        print "\taverage counts as a function of tau"

figC = plt.figure() ### rate of zero-lag coincs
#axC = figC.add_axes(axpos1)
#axCv = figC.add_axes(axpos2)
axC = figC.add_axes(axpos)
axCv = axC.twinx()

figCp = plt.figure() ### rate of slide-coincs with zero-lag
#axCp = figCp.add_axes(axpos1)
#axCpv = figCp.add_axes(axpos2)
axCp = figCp.add_axes(axpos)
axCpv = axCp.twinx()

figCm = plt.figure() ### rate of slide-coincs wiout zero-lag
#axCm = figCm.add_axes(axpos1)
#axCmv = figCm.add_axes(axpos2)
axCm = figCm.add_axes(axpos)
axCmv = axCm.twinx()

### plot observed data
means = np.array( [np.mean([datum[tau]["num_C"]/dur for datum in data]) for tau in taus] )
#print np.polyfit(taus, means, 0)[-1]
stdvs = np.array( [np.std([datum[tau]["num_C"]/dur for datum in data]) for tau in taus] )
#print np.polyfit(taus, stdvs/taus, 3)
axC.plot( taus, means, ".-", color="b")
axC.fill_between( taus, means-stdvs, means+stdvs, color="b", alpha=0.25)
axCv.plot( taus, stdvs, ".-", color="r")

means = np.array( [np.mean([datum[tau]["num_Cp"]/datum[tau]["slideDur"] for datum in data]) for tau in taus] )
#print np.polyfit(taus, means/taus, 0)[-1]
stdvs = np.array( [np.std([datum[tau]["num_Cp"]/datum[tau]["slideDur"] for datum in data]) for tau in taus] )
#print np.polyfit(taus, stdvs/taus, 3)
axCp.plot( taus, means, ".-", color="b")
axCp.fill_between( taus, means-stdvs, means+stdvs, color="b", alpha=0.25)
axCpv.plot( taus, stdvs, ".-", color="r")

means = np.array( [np.mean([datum[tau]["num_Cm"]/datum[tau]["slideDur"] for datum in data]) for tau in taus] )
#print np.polyfit(taus, means/taus, 3)[-1]
stdvs = np.array( [np.std([datum[tau]["num_Cm"]/datum[tau]["slideDur"] for datum in data]) for tau in taus] )
#print np.polyfit(taus, stdvs/taus, 3)
axCm.plot( taus, means, ".-", color="b")
axCm.fill_between( taus, means-stdvs, means+stdvs, color="b", alpha=0.25)
axCmv.plot( taus, stdvs, ".-", color="r")

### plot expected values as dashed line
fine_taus = np.linspace(taus[0], taus[-1], 1001)

ylim = axC.get_ylim()
axC.plot( fine_taus, analytics.Nc(rateA, rateB, rateS, fine_taus), color='k', linestyle="--")
axC.plot( fine_taus, analytics.Nc(rateA, rateB, 0.0, fine_taus), color='grey', linestyle=':')
axC.set_ylim(ylim)

ylim = axCv.get_ylim()
var = analytics.Vcc(rateA, rateB, rateS, fine_taus)/dur
axCv.plot( fine_taus, var**0.5, color='k', linestyle='-.')
axCv.set_ylim(ylim)

ylim = axCp.get_ylim()
axCp.plot( fine_taus, analytics.Np(rateA, rateB, rateS, fine_taus), color='k', linestyle="--")
axCp.plot( fine_taus, analytics.Np(rateA, rateB, 0.0, fine_taus), color='grey', linestyle=':')
axCp.set_ylim(ylim)

ylim = axCpv.get_ylim()
var = np.array( [analytics.Vpp(rateA, rateB, rateS, tau, T=1.0, R=data[0][tau]["slideDur"]/dur)/(dur*(data[0][tau]["slideDur"]/dur)**2) for tau in taus] )
axCpv.plot( taus, var**0.5, color='k', linestyle="-.")
axCpv.set_ylim(ylim)

ylim = axCm.get_ylim()
axCm.plot( fine_taus, analytics.Nm(rateA, rateB, rateS, fine_taus), color='k', linestyle='--')
axCm.plot( fine_taus, analytics.Nm(rateA, rateB, 0.0, fine_taus), color='grey', linestyle=':')
axCm.set_ylim(ylim)

ylim = axCm.get_ylim()
var = np.array( [analytics.Vmm(rateA, rateB, rateS, tau, T=1.0, R=data[0][tau]["slideDur"]/dur)/(dur*(data[0][tau]["slideDur"]/dur)**2) for tau in taus] )
axCmv.plot( taus, var**0.5, color='k', linestyle="-.")
axCm.set_ylim(ylim)

### label
axC.set_xlabel("$\\tau$")
axC.set_ylabel("$\lambda_C$", color='b')

#axCv.set_xlabel("$\\tau$")
axCv.set_ylabel("$\sigma_{\lambda_C}$", color='r')
axCv.yaxis.tick_right()
axCv.yaxis.set_label_position("right")

axCp.set_xlabel("$\\tau$")
axCp.set_ylabel("$\lambda_+$", color='b')

#axCpv.set_xlabel("$\\tau$")
axCpv.set_ylabel("$\sigma_{\lambda_+}$", color='r')
axCpv.yaxis.tick_right()
axCpv.yaxis.set_label_position("right")

axCm.set_xlabel("$\\tau$")
axCm.set_ylabel("$\lambda_-$", color='b')

#axCmv.set_xlabel("$\\tau$")
axCmv.set_ylabel("$\sigma_{\lambda_-}$", color='r')
axCmv.yaxis.tick_right()
axCmv.yaxis.set_label_position("right")

### decorate
axC.grid(opts.grid)
axCp.grid(opts.grid)
axCm.grid(opts.grid)

axCv.grid(opts.grid)
axCpv.grid(opts.grid)
axCmv.grid(opts.grid)

#axC.set_yscale('log')
#axC.set_xscale('log')
#axCv.set_yscale('log')
#axCv.set_xscale('log')

#axCp.set_yscale('log')
#axCp.set_xscale('log')
#axCpv.set_yscale('log')
#axCpv.set_xscale('log')

#axCm.set_yscale('log')
#axCm.set_ylim(ymin=2*1e-3*0.1*0.1, ymax=0.020)
#axCm.set_xscale('log')
#axCmv.set_yscale('log')
#axCmv.set_xscale('log')

### save
figname = "%s/aveC-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t\t", figname
figC.savefig(figname)
plt.close(figC)

figname = "%s/aveCp-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t\t", figname
figCp.savefig(figname)
plt.close(figCp)

figname = "%s/aveCm-tau%s.png"%(opts.output_dir, opts.tag)
if opts.verbose:
        print "\t\t", figname
figCm.savefig(figname)
plt.close(figCm)

#========================
# scatter plots
#========================
if opts.verbose:
	print "\tscatter plots"

extractors = [ ("d_A", lambda x, tau: len(x["A"])+len(x["S"])),\
               ("d_B", lambda x, tau: len(x["B"])+len(x["S"])),\
               ("N_c",lambda x, tau: x[tau]["num_C"]),\
               ("N_+",lambda x, tau: x[tau]["num_Cp"]),\
               ("N_-",lambda x, tau: x[tau]["num_Cm"])\
             ]

means = { \
	"d_A": lambda rA, rB, rS, t, T, R: analytics.dA(rA, rS, T=T),\
	"d_B": lambda rA, rB, rS, t, T, R: analytics.dB(rB, rS, T=T),\
	"N_c":lambda rA, rB, rS, t, T, R: analytics.Nc(rA, rB, rS, t, T=T),\
	"N_+":lambda rA, rB, rS, t, T, R: analytics.Np(rA, rB, rS, t, T=T, R=R),\
	"N_-":lambda rA, rB, rS, t, T, R: analytics.Nm(rA, rB, rS, t, T=T, R=R)\
        }

variances = { \
	"d_A": lambda rA, rB, rS, t, T, R: analytics.Vaa(rA, rS, T=T),\
	"d_B": lambda rA, rB, rS, t, T, R: analytics.Vbb(rB, rS, T=T),\
	"N_c":lambda rA, rB, rS, t, T, R: analytics.Vcc(rA, rB, rS, t, T=T),\
	"N_+":lambda rA, rB, rS, t, T, R: analytics.Vpp(rA, rB, rS, t, T=T, R=R),\
	"N_-":lambda rA, rB, rS, t, T, R: analytics.Vmm(rA, rB, rS, t, T=T, R=R)\
            }

covariances = { \
	("d_A","d_B")  : lambda rA, rB, rS, t, T, R: analytics.Vab(rS, T=T),\
	("d_A","N_c") : lambda rA, rB, rS, t, T, R: analytics.Vac(rA, rB, rS, t, T=T),\
	("d_A","N_+") : lambda rA, rB, rS, t, T, R: analytics.Vap(rA, rB, rS, t, T=T, R=R),\
	("d_A","N_-") : lambda rA, rB, rS, t, T, R: analytics.Vam(rA, rB, rS, t, T=T, R=R),\
	("d_B","N_c") : lambda rA, rB, rS, t, T, R: analytics.Vbc(rA, rB, rS, t, T=T),\
	("d_B","N_+") : lambda rA, rB, rS, t, T, R: analytics.Vbp(rA, rB, rS, t, T=T, R=R),\
	("d_B","N_-") : lambda rA, rB, rS, t, T, R: analytics.Vbm(rA, rB, rS, t, T=T, R=R),\
	("N_c","N_+"): lambda rA, rB, rS, t, T, R: analytics.Vcp(rA, rB, rS, t, T=T, R=R),\
	("N_c","N_-"): lambda rA, rB, rS, t, T, R: analytics.Vcm(rA, rB, rS, t, T=T, R=R),\
	("N_+","N_-"): lambda rA, rB, rS, t, T, R: analytics.Vpm(rA, rB, rS, t, T=T, R=R) \
              }

for tau in taus:
	R = Rs[tau] ### number of slides

	fig = plt.figure(figsize=(10, 10))

	for i, (l, e) in enumerate(extractors[:-1]):
		d = np.array([e(datum, tau) for datum in data])
		# compute expected means and correlations
		m = means[l](rateA, rateB, rateS, tau, dur, R)
		v = variances[l](rateA, rateB, rateS, tau, dur, R)

		set_label = True
		for I, (L, E) in enumerate(extractors[i+1:]):
			D = np.array([E(datum, tau) for datum in data])

			# compute expected means and correlations
			M = means[L](rateA, rateB, rateS, tau, dur, R)
			V = variances[L](rateA, rateB, rateS, tau, dur, R)
			c = covariances[(l,L)](rateA, rateB, rateS, tau, dur, R)

			p = c/(V*v)**0.5 ### correlation coefficient

			#===========================================================================
			### generate single figure with projected histograms
			#===========================================================================
			fig1 = plt.figure()
			axa = fig1.add_axes([0.12, 0.1, 0.6, 0.6])
			axD = fig1.add_axes([0.12, 0.7, 0.6, 0.25])
			axd = fig1.add_axes([0.72, 0.1, 0.25, 0.6])

			axa.plot(D, d, marker='o', markeredgecolor='none', markerfacecolor='k', markersize=2, linestyle='none', alpha=0.5)
			### add contours
			npts = 1001
			xlim = axa.get_xlim()
			xlim = np.max(np.abs(np.array(xlim)-M))
			xlim = (M-xlim, 0.999*(M+xlim))
			ylim = axa.get_ylim()
			ylim = np.max(np.abs(np.array(ylim)-m))
			ylim = (m-ylim, 0.999*(m+ylim))

			X, Y = np.meshgrid( np.linspace(xlim[0], xlim[1], npts), np.linspace(ylim[0], ylim[1], npts) )
			if opts.continuity_correction:
				Z = (2*np.pi*(V*v-c**2))**-0.5 * np.exp( -0.5*( (X-M+0.5)**2/V + (Y-m+0.5)**2/v - 2*c*(X-M+0.5)*(Y-m+0.5)/(V*v) )/(1-p**2) )
			else:
				Z = (2*np.pi*(V*v-c**2))**-0.5 * np.exp( -0.5*( (X-M)**2/V + (Y-m)**2/v - 2*c*(X-M)*(Y-m)/(V*v) )/(1-p**2) )
			axa.contour(X, Y, Z)

			### plot projected histograms
			nbins = min(max(len(D)/10, 5), xlim[1]-xlim[0])
			N, _, _ = axD.hist( D, nbins, histtype="step", color='k', normed=True)
			x = np.linspace(xlim[0], xlim[1], npts)
			if opts.continuity_correction:
				z = (2*np.pi*V)**-0.5 * np.exp( -0.5*(x-M+0.5)**2/V )
			else:
				z = (2*np.pi*V)**-0.5 * np.exp( -0.5*(x-M)**2/V )
			axD.plot( x, z, color='b' )

			nbins = min(max(len(d)/10, 5), ylim[1]-ylim[0])
			n, _, _ = axd.hist( d, nbins, histtype="step", color='k', normed=True, orientation='horizontal')
			y = np.linspace(ylim[0], ylim[1], npts)
			if opts.continuity_correction:
				z = (2*np.pi*v)**-0.5 * np.exp( -0.5*(y-m+0.5)**2/v )
			else:
				z = (2*np.pi*v)**-0.5 * np.exp( -0.5*(y-m)**2/v )
			axd.plot( z, y, color='b' )

			axa.set_xlabel("$%s$"%L)
			axa.set_ylabel("$%s$"%l)

			axD.set_ylabel("$p(%s)$"%L)
			plt.setp(axD.get_xticklabels(), visible=False)

			axd.set_xlabel("$p(%s)$"%l)
			plt.setp(axd.get_yticklabels(), visible=False)

			axa.set_xlim(xlim)
			axa.set_ylim(ylim)

			axD.set_xlim(xlim)
			axD.set_ylim(ymin=0, ymax=1.1*max(N))

			axd.set_ylim(ylim)
			axd.set_xlim(xmin=0, xmax=1.1*max(n))

			axa.grid(opts.grid)
			axD.grid(opts.grid)
			axd.grid(opts.grid)

			figname = "%s/scatter_%s-%s-tau=%.5f%s.png"%(opts.output_dir, l, L, tau, opts.tag)
			if opts.verbose:
				print "\t\t", figname
			fig1.savefig(figname)
			plt.close(fig1)

			#===========================================================================
			### add to "corner figure
			#===========================================================================
			ax = plt.subplot(4, 4, 5*i+I+1)
#			ax.plot(D-M, d-m, marker='o', markeredgecolor='k', markerfacecolor='none', markersize=2, linestyle='none')
			ax.plot(D-M, d-m, marker='o', markeredgecolor='none', markerfacecolor='k', markersize=2, linestyle='none', alpha=0.5)

			### set up contour sample points
			npts = 1001
			xlim = np.max(np.abs(ax.get_xlim()))
			ylim = np.max(np.abs(ax.get_ylim()))

			X, Y = np.meshgrid( np.linspace(-xlim, xlim, npts), np.linspace(-ylim, ylim, npts) )
			
			### compute gaussian distribution
			if opts.continuity_correction: ### apply a continuity correction derived for binomial distributions to the Gaussian approximations. Z(X,Y) -> Z(X+0.5,Y+0.5)
				Z = (2*np.pi*(V*v-c**2))**-0.5 * np.exp( -0.5*( (X+0.5)**2/V + (Y+0.5)**2/v - 2*c*(X+0.5)*(Y+0.5)/(V*v) )/(1-p**2) )
			else:
				Z = (2*np.pi*(V*v-c**2))**-0.5 * np.exp( -0.5*( X**2/V + Y**2/v - 2*c*X*Y/(V*v) )/(1-p**2) )


			### plot contours for gaussian distribution
			ax.contour(X, Y, Z)

			### decorate
			if set_label:
				ax.set_xlabel("$%s - \left<%s\\right>$"%(L,L))
			else:
				plt.setp(ax.xaxis.get_ticklabels(), visible=False)
			plt.setp(ax.yaxis.get_ticklabels(), visible=False)
			set_label = False

			ax.grid(opts.grid)

			ax.set_xlim(-xlim, xlim)
			ax.set_ylim(-ylim, ylim)

		ax.set_ylabel("$%s - \left<%s\\right>$"%(l,l))
		ax.yaxis.set_label_position("right")
		ax.yaxis.tick_right()

	plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

	figname = "%s/scatter-tau=%.5f%s.png"%(opts.output_dir, tau, opts.tag)
	if opts.verbose:
		print "\n\t\t", figname, "\n"
	fig.savefig(figname)
	plt.close(fig)


#========================
# plots separately for each tau
#========================
if opts.individual_tau:
	if opts.verbose:
		print "\tplots for each tau separately"

	### iterating over tau
	for tauNo, tau in enumerate(taus):
		if opts.verbose:
			print "\t============================================"
			print "\t          tau %d / %d -> %.3f"%(tauNo+1, len(taus), tau)
			print "\t============================================"

		### pull out expected rates
		rC = rateC[tau]
		rCp = rateCp[tau]
		rCm = rateCm[tau]

		### HISTOGRAMS
		if opts.verbose:
			print "\tHistograms"

		nbins = max(Ndata/10, 1)

		### define axes
		figC = plt.figure(figsize=(figwidth,figheight))
		ax1C = figC.add_axes(axpos1)
		ax2C = figC.add_axes(axpos2)
	
		figCp = plt.figure(figsize=(figwidth,figheight))
		ax1Cp = figCp.add_axes(axpos1)
		ax2Cp = figCp.add_axes(axpos2)

		figCm = plt.figure(figsize=(figwidth,figheight))
		ax1Cm = figCm.add_axes(axpos1)
		ax2Cm = figCm.add_axes(axpos2)

		### histogram
		ax1C.hist([datum[tau]["num_C"] for datum in data], nbins, histtype="step")
		ax2C.hist([(datum[tau]["num_C"] - rC*dur)/(rC*dur)**0.5 for datum in data], nbins, histtype="step")

		ax1Cp.hist([datum[tau]["num_Cp"] for datum in data], nbins, histtype="step")
		ax2Cp.hist([(datum[tau]["num_Cp"] - rCp*datum[tau]["slideDur"])/(rCp*datum[tau]["slideDur"])**0.5 for datum in data], nbins, histtype="step")

		ax1Cm.hist([datum[tau]["num_Cm"] for datum in data], nbins, histtype="step")
		ax2Cm.hist([(datum[tau]["num_Cm"] - rCm*datum[tau]["slideDur"])/(rCm*datum[tau]["slideDur"])**0.5 for datum in data], nbins, histtype="step")

		### label
		ax1C.set_xlabel("No. C")
		ax2C.set_xlabel("$z_C$")

		ax1Cp.set_xlabel("No. C$_+$")
		ax2Cp.set_xlabel("$z_{C_+}$")

		ax1Cm.set_xlabel("No. C$_-$")
		ax2Cm.set_xlabel("$z_{C_-}$")

		ax1C.set_ylabel("count")
		ax2C.set_ylabel("count")
		ax2C.yaxis.tick_right()
		ax2C.yaxis.set_label_position("right")

		ax1Cp.set_ylabel("count")
		ax2Cp.set_ylabel("count")
		ax2Cp.yaxis.tick_right()
		ax2Cp.yaxis.set_label_position("right")

		ax1Cm.set_ylabel("count")
		ax2Cm.set_ylabel("count")
		ax2Cm.yaxis.tick_right()
		ax2Cm.yaxis.set_label_position("right")

		### decorate
		ax1C.grid(opts.grid)
		ax2C.grid(opts.grid)

		ax1Cp.grid(opts.grid)
		ax2Cp.grid(opts.grid)

		ax1Cm.grid(opts.grid)
		ax2Cm.grid(opts.grid)

		### save
		figname = "%s/C-tau_%.5f%s.png"%(opts.output_dir, tau, opts.tag)
		if opts.verbose:
			print "\t\t", figname
		figC.savefig(figname)
		plt.close(figC)

		figname = "%s/Cp-tau_%.5f%s.png"%(opts.output_dir, tau, opts.tag)
		if opts.verbose:
			print "\t\t", figname
		figCp.savefig(figname)
		plt.close(figCp)

		figname = "%s/Cm-tau_%.5f%s.png"%(opts.output_dir, tau, opts.tag)
		if opts.verbose:
			print "\t\t", figname
		figCm.savefig(figname)
		plt.close(figCm)

		### SCATTER PLOTS
		if opts.verbose:
			print "\tScatter Plots"

		keys = "num_C num_Cp num_Cm".split()
		### against individual streams
		for key1 in "S A B".split():
			val1 = [len(datum[key1]) for datum in data]

			for key2 in keys:
				val2 = [datum[tau][key2] for datum in data]

				fig = plt.figure()
				ax = fig.add_axes(axpos)

				### plot
				ax.plot(val1, val2, marker="o", linestyle="none")

				### label
				ax.set_xlabel(key1)
				ax.set_ylabel("%s tau=%.5f"%(key2, tau))

				### decorate
				ax.grid(opts.grid)

				### save
				figname = "%s/%s-%s_tau-%.5f%s.png"%(opts.output_dir, key1, key2, tau, opts.tag)
				if opts.verbose:
					print "\t\t", figname
				fig.savefig(figname)
				plt.close(fig)


		### pairs of coincs
		for ind1, key1 in enumerate(keys[:-1]):
			val1 = [datum[tau][key1] for datum in data]

			for key2 in keys[ind1+1:]:
				val2 = [datum[tau][key2] for datum in data]

				fig = plt.figure()
				ax = fig.add_axes(axpos)

				### plot
				ax.plot(val1, val2, marker="o", markerfacecolor="none", linestyle="none")

				### label
				ax.set_xlabel("%s tau=%.5f"%(key1, tau))
				ax.set_ylabel("%s tau=%.5f"%(key2, tau))

				### decorate
				ax.grid(opts.grid)

				### save
				figname = "%s/%s_tau-%.5f-%s_tau-%.5f%s.png"%(opts.output_dir, key1, tau, key2, tau, opts.tag)
				if opts.verbose:
					print "\t\t", figname
				fig.savefig(figname)

