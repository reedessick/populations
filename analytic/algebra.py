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

N= sympy.symbols("N")
### N = T/(2*t)
R = N-1

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
Vpp = N*R * ( (ea+es)*(eb+es) + es**2 
                  + R * (ea+es)*(eb+es)*(ea + eb + 4*es) )

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
Vmm = N*R * ea*eb * exp(-ea)*exp(-eb)*exp(-2*es) \
	* ( 1 + ea*(1-exp(-eb)*exp(-es)) + eb*(1-exp(-ea)*exp(-es)) \
	    + ea*eb*(1-exp(-eb)*exp(-es))*(1-exp(-ea)*exp(-es)) + ea*eb* exp(-ea)*exp(-eb)*exp(-2*es) \
	    + R * ( eb*exp(-ea)*exp(-es) + ea*exp(-eb)*exp(-es) \
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

Vap = N*R * (eb + 2*es)*(ea+es)

### check that we've got this correct
assert sympy.expand(Vap) == sympy.expand(Nap-Na*Np), "failed consistency check for Vap"

#========================

Nbp = N*(N-1) * ( (eb2 + 2*eb*es + es2)*(ea+es) + (eb+es)*(ea*eb + es*(ea+eb) + es2) ) \
      + N*(N-1)*(N-2) * ( (ea+es) * (eb+es)**2 )

Vbp = N*R * (ea+2*es)*(eb+es)

### check that we've got this correct
assert sympy.expand(Vbp) == sympy.expand(Nbp-Nb*Np), "failed consistency check for Vbp"

#========================

Nam = N*(N-1) * ea2*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es) \
      + N*(N-1)*(N-2) * (ea+es) * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es)

Vam = N*R * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * (1 - ea - 2*es )

### check that we've got this correct
assert sympy.expand(Vam) == sympy.expand(Nam-Na*Nm), "failed consistency check for Vam"

#========================

Nbm = N*(N-1) * eb2*exp(-ea)*exp(-es) * ea*exp(-eb)*exp(-es) \
      + N*(N-1)*(N-2) * (eb+es) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vbm = N*R * eb*ea*exp(-eb)*exp(-ea)*exp(-2*es) * (1 - eb - 2*es )

### check that we've got this correct
assert sympy.expand(Vbm) == sympy.expand(Nbm-Nb*Nm), "failed consistency check for Vbm"

#========================

Ncp = N*(N-1) * ( (ea2*eb + 2*ea*eb*es + eb*es2 + ea2*es + 2*ea*es2 + es3)*(eb+es) \
                 + (eb2*ea + 2*eb*ea*es + ea*es2 + eb2*es + 2*eb*es2 + es3)*(ea+es) ) \
      + N*(N-1)*(N-2) * (ea*eb + es*(ea+eb) + es2) * (ea+es) * (eb+es)

Vcp = N*R * ( (ea*eb + es*(ea+eb + es))*(ea+eb+4*es) + es*(ea+eb+2*es) )

### check that we've got this correct
assert sympy.expand(Vcp) == sympy.expand(Ncp-Nc*Np), "failed consistency check for Vcp"

#========================

Ncm = N*(N-1)*(N-2) * (ea*eb + es*(ea+eb) + es2) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vcm = -2*N*R * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * (ea*eb + es*(ea+eb+es) + es)

### check that we've got this correct
assert sympy.expand(Vcm) == sympy.expand(Ncm-Nc*Nm), "failed consistency check for Vcm"

#========================

Npm = N*(N-1) * ea2*exp(-eb)*exp(-es) * eb2*exp(-ea)*exp(-es) \
      + N*(N-1)*(N-2) * ( ea2*exp(-eb)*exp(-es) * (eb+es)*eb*exp(-ea)*exp(-es) \
                         + eb2*exp(-ea)*exp(-es) * (ea+es)*ea*exp(-eb)*exp(-es) ) \
      + N*(N-1)*(N-2)*(N-3) * (ea+es) * (eb+es) * ea*exp(-eb)*exp(-es) * eb*exp(-ea)*exp(-es)

Vpm = N*R * ea*eb*exp(-ea)*exp(-eb)*exp(-2*es) * ( (1+ea)*(1+eb) + 2*(ea+es)*(eb+es) \
                                                        - (1+ea)*(eb+es) - (ea+es)*(1+eb) \
                                                       + R * ((1+ea)*(eb+es) + (ea+es)*(1+eb) \
                                                                   - 4*(ea+es)*(eb+es)) \
                                                     )

### check that we've got this correct
assert sympy.expand(Vpm) == sympy.expand(Npm-Np*Nm), "failed consistency check for Vpm"

#=================================================
#
# covariances between different taus
#
#=================================================
t1, t2 = sympy.symbols("tau_1 tau_2")
### t1 <= t2

N = sympy.symbols("N_1")
### N = T/(2*t1)
R = N-1

z = t2/t1
### z = N/n >= 1 -> N >= n
### z >= 1

n = N/z
### n = T/(2*t2)
r = N-1

e1a = 2*t1*ra
e1b = 2*t1*rb
e1s = 2*t1*rs

e1a2 = e1a + e1a**2
e1b2 = e1b + e1b**2
e1s2 = e1s + e1s**2

e1s3 = e1s + 3*e1s**2 + e1s**3
e1s4 = e1s + 7*e1s**2 + 6*e1s**3 + e1s**4

e2a = 2*t2*ra
e2b = 2*t2*rb
e2s = 2*t2*rs

e2a2 = e2a + e2a**2
e2b2 = e2b + e2b**2
e2s2 = e2s + e2s**2

e2s3 = e2s + 3*e2s**2 + e2s**3
e2s4 = e2s + 7*e2s**2 + 6*e2s**3 + e2s**4

#========================
### first moments
#========================

Na1 = N * (e1a + e1s)
Na2 = n * z * (e1a + e1s) 

assert sympy.expand(Na2) == sympy.expand(n * (e2a + e2s)), "failed consistency check for Na2"

Nb1 = N * (e1b + e1s)
Nb2 = n * z * (e1b + e1s)

assert sympy.expand(Nb2) == sympy.expand(n * (e2b + e2s)), "failed consistency check for Nb2"

Nc1 = N * (e1a*e1b + e1a*e1s + e1b*e1s + e1s2)
Nc2 = n * z * (e1a*e1b + e1a*e1s + e1b*e1s + e1s2) + n * z*(z-1) * (e1a+e1s)*(e1b+e1s)

assert sympy.expand(Nc2) == sympy.expand(n * (e2a*e2b + e2a*e2s + e2b*e2s + e2s2)), "failed consistency check for Nc2"

Np1 = N*(N-1) * (e1a + e1s)*(e1b + e1s)
Np2 = n*(n-1) * z*(e1a + e1s) * z*(e1b + e1s)

assert sympy.expand(Np2) == sympy.expand(n*(n-1) * (e2a + e2s)*(e2b + e2s)), "failed consistency check for Np2"

Nm1 = N*(N-1) * e1a*e1b * exp(-e1a)*exp(-e1b)*exp(-2*e1s)
Nm2 = n*(n-1) * z*e1a*exp(-z*e1b)*exp(-z*e1s) * z*e1b*exp(-z*e1a)*exp(-z*e1s)

assert sympy.expand(Nm2) == sympy.expand(n*(n-1) * e2a*e2b * exp(-e2a-e2b-2*e2s)), "failed consistency check for Nm2"

#========================
### second moments
#========================

Nc1c2 = n * z *(e1a2*e1b2 + 2*e1a2*e1b*e1s + e1a2*e1s2 + 2*e1a*e1b2*e1s + 4*e1a*e1b*e1s2 + 2*e1a*e1s3 + e1b2*e1s2 + 2*e1b*e1s3 + e1s4) \
        + n * z*(z-1) * ( (e1a2*e1b + 2*e1a*e1b*e1s + e1b*e1s2 + e1a2*e1s + 2*e1a*e1s2 + e1s3)*(e1b+e1s) \
                          + (e1a+e1s)*(e1a*e1b2 + 2*e1a*e1b*e1s + e1a*e1s2 + e1b*e1s + 1*e1b*e1s2 + e1s3) ) \
        + n * z*(z-1)*(z-2) * (e1a*e1b + e1a*e1s + e1b*e1s + e1s2)*(e1a+e1s)*(e1b+e1s) \
        + n*(n-1) * z * (e1a*e1b + e1a*e1s + e1b*e1s + e1s2) * ( z * (e1a*e1b + e1a*e1s + e1b*e1s + e1s2) + z*(z-1) * (e1a+e1s)*(e1b+e1s) )

Nc1p2 = n*(n-1) * ( (z*(e1a2*e1b + e1a2*e1s + 2*e1a*e1b*e1s + 2*e1a*e1s2 + e1b*e1s2 + e1s3) \
                      + z*(z-1)*(e1a*e1b+e1a*e1s+e1b*e1s+e1s2)*(e1a+e1s))* z*(e1b+e1s) \
                    + (z*(e1b2*e1a + e1b2*e1s + 2*e1b*e1a*e1s + 2*e1b*e1s2 + e1a*e1s2 + e1s3) \
                      + z*(z-1)*(e1b*e1a+e1b*e1s+e1a*e1s+e1s2)*(e1b+e1s))* z*(e1a+e1s) ) \
        + n*(n-1)*(n-2) * z*(e1a*e1b+e1a*e1s+e1b*e1s+e1s2) * z*(e1a+e1s) + z*(e1b+e1s) 

Nc1m2 = n*(n-1)*(n-2) * z*(e1a*e1b + e1a*e1s + e1b*e1s + e1s2) * z*e1a*exp(-z*e1b)*exp(-z*e1s) * z*e1b*exp(-z*e1a)*exp(-z*e1s)

Np1c2 = 0

Np1p2 = 0

Np1m2 = 0

Nm1c2 = 0

Nm1p2 = 0

Nm1m2 = 0

#========================
### Variances
#========================
print "WARNING: multi-tau variances have not been symplified analytically..."

Vc1c2 = Nc1c2 - Nc1*Nc2

Vc1p2 = Nc1p2 - Nc1*Np2

Vc1m2 = Nc1m2 - Nc1*Nm2

Vp1c2 = Np1c2 - Np1*Nc2

Vp1p2 = Np1p2 - Np1*Np2

Vp1m2 = Np1m2 - Np1*Nm2

Vm1c2 = Nm1c2 - Nm1*Nc2

Vm1p2 = Nm1p2 - Nm1*Np2

Vm1m2 = Nm1m2 - Nm1*Nm2

### check that we've got this correct
assert sympy.expand(Vc1c2) == sympy.expand(Nc1c2 - Nc1*Nc2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vc1p2) == sympy.expand(Nc1p2 - Nc1*Np2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vc1m2) == sympy.expand(Nc1m2 - Nc1*Nm2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vp1c2) == sympy.expand(Np1c2 - Np1*Nc2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vp1p2) == sympy.expand(Np1p2 - Np1*Np2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vp1m2) == sympy.expand(Np1m2 - Np1*Nm2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vm1c2) == sympy.expand(Nm1c2 - Nm1*Nc2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vm1p2) == sympy.expand(Nm1p2 - Nm1*Np2), "failed consistency check for Vc1c2"

### check that we've got this correct
assert sympy.expand(Vm1m2) == sympy.expand(Nm1m2 - Nm1*Nm2), "failed consistency check for Vc1c2"

#=================================================
# All consistency passes check, so we can now move on with the approximation of the likelihood!
#=================================================
print "success! you can do algebra!\nnow you have to use these functions to approximate the likelihood"


