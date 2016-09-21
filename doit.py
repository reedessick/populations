usage = "doit.py [--options] dat.pkl"
description = """ attempt to recover the posterior using emcee implementation """

#=================================================

import pickle

import numpy as np
import emcee

import analytics

from optparse import OptionParser

#=================================================

def means_covar(tau, T, R, rateA, rateB, rateS):
    dA = analytics.dA(rateA, rateS, T=T)
    dB = analytics.dB(rateB, rateS, T=T)
    Nc = analytics.Nc(rateA, rateB, rateS, tau, T=T)
    Np = analytics.Np(rateA, rateB, rateS, tau, T=T, R=R)
    Nm = analytics.Nm(rateA, rateB, rateS, tau, T=T, R=R)

    Vaa = analytics.Vaa(rateA, rateS, T=T)
    Vbb = analytics.Vbb(rateB, rateS, T=T)
    Vcc = analytics.Vcc(rateA, rateB, rateS, tau, T=T)
    Vpp = analytics.Vpp(rateA, rateB, rateS, tau, T=T, R=R)
    Vmm = analytics.Vmm(rateA, rateB, rateS, tau, T=T, R=R)

    Vab = analytics.Vab(rateS, T=T)
    Vac = analytics.Vac(rateA, rateB, rateS, tau, T=T)
    Vap = analytics.Vap(rateA, rateB, rateS, tau, T=T, R=R)
    Vam = analytics.Vam(rateA, rateB, rateS, tau, T=T, R=R)

    Vbc = analytics.Vbc(rateA, rateB, rateS, tau, T=T)
    Vbp = analytics.Vbp(rateA, rateB, rateS, tau, T=T, R=R)
    Vbm = analytics.Vbm(rateA, rateB, rateS, tau, T=T, R=R)

    Vcp = analytics.Vcp(rateA, rateB, rateS, tau, T=T, R=R)
    Vcm = analytics.Vcm(rateA, rateB, rateS, tau, T=T, R=R)

    Vpm = analytics.Vpm(rateA, rateB, rateS, tau, T=T, R=R)

    means = np.array([dA, dB, Nc, Np, Nm])
    covar = np.array([[Vaa, Vab, Vac, Vap, Vam],
                      [Vab, Vbb, Vbc, Vbp, Vbm],
                      [Vac, Vbc, Vcc, Vcp, Vcm],
                      [Vap, Vbp, Vcp, Vpp, Vpm],
                      [Vam, Vbm, Vcm, Vpm, Vmm]])

    return means, np.linalg.inv(covar)

def means_covar_ignore_Nc(tau, T, R, rateA, rateB, rateS):
    dA = analytics.dA(rateA, rateS, T=T)
    dB = analytics.dB(rateB, rateS, T=T)
    Np = analytics.Np(rateA, rateB, rateS, tau, T=T, R=R)
    Nm = analytics.Nm(rateA, rateB, rateS, tau, T=T, R=R)

    Vaa = analytics.Vaa(rateA, rateS, T=T)
    Vbb = analytics.Vbb(rateB, rateS, T=T)
    Vpp = analytics.Vpp(rateA, rateB, rateS, tau, T=T, R=R)
    Vmm = analytics.Vmm(rateA, rateB, rateS, tau, T=T, R=R)

    Vab = analytics.Vab(rateS, T=T)
    Vap = analytics.Vap(rateA, rateB, rateS, tau, T=T, R=R)
    Vam = analytics.Vam(rateA, rateB, rateS, tau, T=T, R=R)

    Vbp = analytics.Vbp(rateA, rateB, rateS, tau, T=T, R=R)
    Vbm = analytics.Vbm(rateA, rateB, rateS, tau, T=T, R=R)

    Vpm = analytics.Vpm(rateA, rateB, rateS, tau, T=T, R=R)

    means = np.array([dA, dB, Np, Nm])
    covar = np.array([[Vaa, Vab, Vap, Vam],
                      [Vab, Vbb, Vbp, Vbm],
                      [Vap, Vbp, Vpp, Vpm],
                      [Vam, Vbm, Vpm, Vmm]])

    return means, np.linalg.inv(covar)


def logpost(rates, x, ignore_Nc=False, **kwargs):
    """
    if ignore_Nc:
        x ~ (d_A, d_B, N_p, N_m, tau, T, R)
    else:
        x ~ (d_A, d_B, N_c, N_p, N_m, tau, T, R)
    """
    rateA, rateB, rateC = rates
    like = loglike(x, rateA, rateB, rateC, ignore_Nc=ignore_Nc, **kwargs)
    prior = logprior(rateA, rateB, rateC, **kwargs)

    return like + prior

def loglike(x, rateA, rateB, rateC, ignore_Nc=False, safe=True, **kwargs):
    """
    if ignore_Nc:
        x ~ (d_A, d_B, N_p, N_m, tau, T, R)
    else:
        x ~ (d_A, d_B, N_c, N_p, N_m, tau, T, R)
    """
    if ignore_Nc:
        dx = 7
        mc = means_covar_ignore_Nc
    else:
        dx = 8
        mc = means_covar

    lenx = len(x)
    like = 0.0
    for i in xrange( lenx / dx ):
        y = x[i*dx:(i+1)*dx]
        tau, T, R = y[-3:]
        y = y[:-3]

        means, covar = mc(tau, T, R, rateA, rateB, rateS)
        dy = y - means
        
        like -= 0.5 * np.sum( dy * np.sum( covar * dy, axis=1 ), axis=0 )

    return like

def logprior(rateA, rateB, rateC, **kwargs):
    """
    positive semi-definite priors on rates
    """
    if rateA < 0:
        return -np.infty
    if rateB < 0:
        return -np.infty
    if rateS < 0:
        return -np.infty

    return 0.0

def init_ensemble(x, nwalkers):
    """
    initialize the position of each chain in the ensemble
    """
    ndim = 3 ### always the case

    ### compute what the expected value is for rateA, rateB from dA, dB alone
    ### start rateS at some small value, possibly 0
    ### scatter these according to a gaussian approximation
    ###     need the approximate variance on rateS from upper limits on poisson process (~1/T)
    ### define a set of variances and 

    lenx = len(x)
    if lenx % 7: # include Nc
        ra = np.mean( x[0::8]/x[6::8] )
        rb = np.mean( x[1::8]/x[6::8] )
        rs = np.mean( 1.0/x[6::8] )
    else:
        ra = np.mean( x[0::7]/x[5::7] )
        rb = np.mean( x[1::7]/x[5::7] )
        rs = np.mean( 1.0/x[5::7] )

    pos = np.random.normal(size=(nwalkers, ndim)) ### normally distributed

    pos *= np.array( [ra**0.5, rb**0.5, rs**0.5] )
    pos += np.array( [ra, rb, 0.0] )

    pos = np.abs(pos)

    return pos


#=================================================

parser = OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-n", "--nwalkers", default=1, type="int")
parser.add_option("-a", "--scale", default=2.0, type="float")
parser.add_option("-T", "--threads", default=1, type="int")

parser.add_option("-N", "--nsteps", default=100000, type="int")
parser.add_option("", "--incremental", default=None, type="int")

parser.add_option("", "--ignore-Nc", default=False, action="store_true")

parser.add_option("", "--tau", default=0.1, type="float", help="only use the data in dat.pkl that corresponds to this tau. Implemented as such until we include the multi-tau correlations so that there is only ever one tau measurement per noise instantiation")

parser.add_option("-t", "--time", default=False, action="store_true")

parser.add_option("-o", "--output-filename", default=None, type="string")

opts, args = parser.parse_args()

if len(args) != 1:
    raise ValueError("please supply exactly one dat.pkl file as an argument")
datpkl = args[0]

if opts.time:
    import time

opts.verbose = opts.verbose or opts.time

if not opts.incremental:
    opts.incremental = opts.nsteps

#=================================================

ndim = 3 ### our model has 3 rates in it: lambdaA, lambdaB, lambdaS

#=================================================
### extract data from datpkl

if opts.verbose:
    print "===================================================="
    print "  loading data from %s"%(datpkl)

if opts.time:
    to = time.time()

file_obj = open(datpkl, "r")
params = pickle.load(file_obj)
data = pickle.load(file_obj)
file_obj.close()

T = params['dur']
Ndata = len(data)

if opts.time:
    print "    %.6f sec"%(time.time()-to)

#=================================================
### set up data vectors, likelihood, etc

if opts.verbose:
    print "===================================================="
    print "  setting up data vector"

if opts.time:
    to = time.time()

x = []
if opts.ignore_Nc:
    for datum in data:
        if datum.has_key(opts.tau):
            dA = len(datum['A'])
            dB = len(datum['B'])
            Np = datum[opts.tau]['num_Cp']
            Nm = datum[opts.tau]['num_Cm']
            R = datum[opts.tau]['slideDur'] / T
            x += [dA, dB, Np, Nm, opts.tau, T, R]
else:
    for datum in data:
        if datum.has_key(opts.tau):
            dA = len(datum['A'])
            dB = len(datum['B'])
            Nc = datum[opts.tau]['num_C']
            Np = datum[opts.tau]['num_Cp']
            Nm = datum[opts.tau]['num_Cm']
            R = datum[opts.tau]['slideDur'] / T
            x += [dA, dB, Nc, Np, Nm, opts.tau, T, R]
   
x = np.array(x)

if opts.time:
    print "    %.6f sec"%(time.time()-to)
 
#=================================================
### set up IC

### initialize points in parameter space
if opts.verbose:
    print "==================================================="
    print "  initializing walker positions"

if opts.time:
    to = time.time()

p0 = init_ensemble(x, opts.nwalkers)
#p0 = np.random.rand(ndim * opts.nwalkers).reshape((opts.nwalkers, ndim))

### instantiate sampler
#rateA = np.zeros(opts.nwalkers)
#rateB = np.zeros(opts.nwalkers)
#rateS = np.zeros(opts.nwalkers)
rateA = np.mean(p0[:,0])
rateB = np.mean(p0[:,1])
rateS = np.mean(p0[:,2])

if opts.time:
    print "    %.6f sec"%(time.time()-to)

#=================================================
### set up sampler

if opts.verbose:
    print "==================================================="
    print "  instantiating EnsembleSampler"

if opts.time:
    to = time.time()

sampler = emcee.EnsembleSampler(opts.nwalkers, ndim, logpost, \
            a=opts.scale, args=[x], kwargs={"ignore_Nc":opts.ignore_Nc}, threads=opts.threads)

if opts.time:
    print "    %.6f sec"%(time.time()-to)

#=================================================
### run sampler

if opts.verbose:
    print "==================================================="

niters = (opts.nsteps / opts.incremental)
steps = 0
pos = p0
for count in xrange( opts.nsteps / opts.incremental ):
    if opts.verbose:
        print "  running sampler for %d steps -> (%d + %d) / %d "%(opts.incremental, steps, opts.incremental, opts.nsteps)

    if opts.time:
        to = time.time()

    pos, prob, state = sampler.run_mcmc(pos, opts.incremental)

    if opts.time:
        print "    %.6f sec"%(time.time()-to)

    steps += opts.incremental

if steps < opts.nsteps:
    if opts.verbose:
        print "  running sampler for %d steps -> (%d + %d) / %d"%(opts.nsteps - steps, steps, opts.nsteps-steps, opts.nsteps)

    if opts.time:
        to = time.time()

    pos, prob, state = sampler.run_mcmc(pos, opts.nsteps-steps)

    if opts.time:
        print "    %.6f sec"%(time.time()-to)

#=================================================
### report

if opts.verbose:
    print "==================================================="
    print "  reporting chain"

if opts.time:
    to = time.time()

if opts.output_filename:
    file_obj = open(opts.output_filename, "w")
else:
    import sys
    file_obj = sys.stdout

for s in xrange(opts.nsteps):
    print >> file_obj, " ".join([str(sample) for sample in sampler.chain[:,s,:].flatten()])

if opts.output_filename:
    file_obj.close()

if opts.time:
    print "    %.6f sec"%(time.time()-to)
