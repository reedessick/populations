usage="python algebra.py"
description="written to compute some algebra for the moments of rate statistics"
author = "R. Essick (reed.essick@ligo.org)"

#=================================================

import sympy
from sympy import exp

#=================================================
### define relevant variables

ra, rb, rs = sympy.symbols("lambda_1 lambda_2 lambda_0")

T, t = sympy.symbols("T tau")

N, R = sympy.symbols("N R")
### N = T/(2*t)
### R = N-1

ea = 2*t*ra
eb = 2*t*rb
es = 2*t*rs

ea2 = ea + ea**2
eb2 = eb + eb**2
es2 = es + es**2

es3 = es + 3*es**2 + es**3
es4 = es + 7*es**2 + 6*es**3 + es**4

#=================================================
# dA
#=================================================
Na = N * (ea + es)

Vaa = N * (ea + es)

#=================================================
# dB
#=================================================
Nb = N * (eb + es)

Vbb = N * (eb + es)

#=================================================
# Nc
#=================================================
Nc  = N * (ea*eb + es*(ea+eb) + es2)

Nc2 = N * (ea2*eb2 + 2*ea2*eb*es + ea2*es2 + 2*ea*eb2*es + 4*ea*eb*es2 + 2*ea*es3 + eb2*es2 + 2*eb*es3 + es4 ) \
      + N*(N-1) * (ea*eb + es*(ea+eb+es) + es)**2

### from the write-up
Vcc = N * (es + ea*eb + 3*es*(ea+eb+2*es) + (ea+eb+4*es)*(ea*eb + es*(ea+eb+es))) 

### check that we've got this correct
assert sympy.expand(Vcc) == sympy.expand(Nc2-Nc**2), "failed consistency check for Vcc"

#=================================================
# Np
#=================================================
Np  = N*(N-1) * (ea + es)*(eb + es)

Np2 = N*(N-1) * ( (ea2 + 2*ea*es + es2)*(eb2 + 2*eb*es + es2) + (ea*eb + es*(ea+eb) + es2)**2 ) \
      + N*(N-1)*(N-2) * ( (ea2 + 2*ea*es + es2)*(eb+es)**2 + (eb2 + 2*eb*es + es2)*(ea+es)**2 \
                          + 2*(ea*eb + es*(ea+eb) + es2)*(ea+es)*(eb+es) ) \
      + N*(N-1)*(N-2)*(N-3) * (ea+es)**2 * (eb+es)**2 

### from the write-up
Vpp = N*(N-1) * ( (ea+es)*(eb+es) + es**2 
                  + (N-1) * (ea+es)*(eb+es)*(ea + eb + 4*es) )

### check that we've got this correct
assert sympy.expand(Vpp) == sympy.expand(Np2-Np**2), "failed consistency check for Vpp"

#=================================================
# Nm
#=================================================
Nm  = N*(N-1) * ea*eb * exp(-ea-eb-2*es)

Nm2 = N*(N-1) * ea2*eb2 * exp(-ea)*exp(-eb)*exp(-2*es) \
    + N*(N-1)*(N-2) * ( ea2*eb**2 * exp(-2*ea)*exp(-eb)*exp(-3*es) + ea**2*eb2*exp(-ea)*exp(-2*eb)*exp(-3*es) ) \
    + N*(N-1)*(N-2)*(N-3) * ea**2*eb**2 * exp(-2*ea)*exp(-2*eb)*exp(-4*es)

### from the write-up
Vmm = N*(N-1) * ea*eb * exp(-ea)*exp(-eb)*exp(-2*es) \
	* ( 1 + ea*(1-exp(-eb)*exp(-es)) + eb*(1-exp(-ea)*exp(-es)) \
	    + ea*eb*(1-exp(-eb)*exp(-es))*(1-exp(-ea)*exp(-es)) + ea*eb* exp(-ea)*exp(-eb)*exp(-2*es) \
	    + (N-1) * ( eb*exp(-ea)*exp(-es) + ea*exp(-eb)*exp(-es) \
	                + ea*eb*(exp(-ea)*exp(-es) + exp(-eb)*exp(-es) - 4*exp(-ea)*exp(-eb)*exp(-2*es) ) ) )

### check that we've got this correct
assert sympy.expand(Vmm) == sympy.expand(Nm2-Nm**2), "failed consistency check for Vmm"

#=================================================
# covariances
#=================================================
Nab = N * (ea*eb + ea*es + eb*es + es2) \
      + N*(N-1) * (ea+es)*(eb+es)

Vab = N * es

### check that we've go this correct
assert sympy.expand(Vab) == sympy.expand(Nab-Na*Nb), "failed consistency check for Vab"

#========================

Nac = N * ( ea2*(eb+es) + 2*ea*eb*es + 2*ea*es2 + eb*es2 + es3 ) \
      + N*(N-1) * (ea+es)*(ea*eb + es*(ea+eb) + es2)

Vac = N * ( es + (ea+es)*(eb + 2*es) )

### check that we've got this correct
assert sympy.expand(Vac) == sympy.expand(Nac-Na*Nc), "failed consistency check for Vac"

#========================

Nbc = N * ( eb2*(ea+es) + 2*eb*ea*es + 2*eb*es2 + ea*es2 + es3 ) \
      + N*(N-1) * (eb+es)*(ea*eb + es*(ea+eb) + es2)

Vbc = N * ( es + (eb+es)*(ea + 2*es) )

### check that we've got this correct
assert sympy.expand(Vbc) == sympy.expand(Nbc-Nb*Nc), "failed consistency check for Vbc"

#========================

Nap = N*(N-1) * ( (ea2 + 2*ea*es + es2)*(eb+es) + (ea+es)*(ea*eb + es*(ea+eb) + es2) ) \
      + N*(N-1)*(N-2) * ( (ea+es)**2 * (eb+es) )

Vap = N*(N-1) * (eb + 2*es)*(ea+es)

### check that we've got this correct
assert sympy.expand(Vap) == sympy.expand(Nap-Na*Np), "failed consistency check for Vap"

#========================

Nbp = N*(N-1) * ( (eb2 + 2*eb*es + es2)*(ea+es) + (eb+es)*(ea*eb + es*(ea+eb) + es2) ) \
      + N*(N-1)*(N-2) * ( (ea+es) * (eb+es)**2 )

Vbp = N*(N-1) * (ea+2*es)*(eb+es)

### check that we've got this correct
assert sympy.expand(Vbp) == sympy.expand(Nbp-Nb*Np), "failed consistency check for Vbp"

#========================

Nam = N*(N-1) * ea2*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es) \
      + N*(N-1)*(N-2) * (ea+es) * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es)

Vam = N*(N-1) * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * (1 - ea - 2*es )

### check that we've got this correct
assert sympy.expand(Vam) == sympy.expand(Nam-Na*Nm), "failed consistency check for Vam"

#========================

Nbm = N*(N-1) * eb2*exp(-ea)*exp(-es) * ea*exp(-eb)*exp(-es) \
      + N*(N-1)*(N-2) * (eb+es) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vbm = N*(N-1) * eb*ea*exp(-eb)*exp(-ea)*exp(-2*es) * (1 - eb - 2*es )

### check that we've got this correct
assert sympy.expand(Vbm) == sympy.expand(Nbm-Nb*Nm), "failed consistency check for Vbm"

#========================

Ncp = N*(N-1) * ( (ea2*eb + 2*ea*eb*es + eb*es2 + ea2*es + 2*ea*es2 + es3)*(eb+es) \
                 + (eb2*ea + 2*eb*ea*es + ea*es2 + eb2*es + 2*eb*es2 + es3)*(ea+es) ) \
      + N*(N-1)*(N-2) * (ea*eb + es*(ea+eb) + es2) * (ea+es) * (eb+es)

Vcp = N*(N-1) * ( (ea*eb + es*(ea+eb + es))*(ea+eb+4*es) + es*(ea+eb+2*es) )

### check that we've got this correct
assert sympy.expand(Vcp) == sympy.expand(Ncp-Nc*Np), "failed consistency check for Vcp"

#========================

Ncm = N*(N-1)*(N-2) * (ea*eb + es*(ea+eb) + es2) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vcm = -2*N*(N-1) * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * (ea*eb + es*(ea+eb+es) + es)

### check that we've got this correct
assert sympy.expand(Vcm) == sympy.expand(Ncm-Nc*Nm), "failed consistency check for Vcm"

#========================

Npm = N*(N-1) * ea2*exp(-eb)*exp(-es) * eb2*exp(-ea)*exp(-es) \
      + N*(N-1)*(N-2) * ( ea2*exp(-eb)*exp(-es) * (eb+es)*eb*exp(-ea)*exp(-es) \
                         + eb2*exp(-ea)*exp(-es) * (ea+es)*ea*exp(-eb)*exp(-es) ) \
      + N*(N-1)*(N-2)*(N-3) * (ea+es) * (eb+es) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vpm = N*(N-1) * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * ( (1+ea)*(1+eb) + 2*(ea+es)*(eb+es) \
                                                        - (1+ea)*(eb+es) - (ea+es)*(1+eb) \
                                                       + (N-1) * ((1+ea)*(eb+es) + (ea+es)*(1+eb) \
                                                                   - 4*(ea+es)*(eb+es)) \
                                                     )

### check that we've got this correct
assert sympy.expand(Vpm) == sympy.expand(Npm-Np*Nm), "failed consistency check for Vpm"

#=================================================
# All consistency passes check, so we can now move on with the approximation of the likelihood!
#=================================================
print "success! you can do algebra!\nnow you have to use these functions to approximate the likelihood"


