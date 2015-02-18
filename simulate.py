#!/usr/bin/python
usage = """simulate.py [--options] """
description = """written to efficiently simulate the number of samples observed via timeslides """

import os
import pickle
from copy import deepcopy

import series_utils as su
np = su.np

from optparse import OptionParser

#=================================================

parser=OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("", "--dur", default=1e4, type="float")

parser.add_option("", "--rateS", default=1e-3, type="float")

parser.add_option("", "--rateA", default=0.1, type="float")
parser.add_option("", "--rateB", default=0.1, type="float")

parser.add_option("", "--tau", default=[0.01], type="float", action="append", help="tau used for clustering and coincidence")

parser.add_option("", "--slide-type", default="cyclic", type="string", help="either 'cyclic' or 'linear'")

parser.add_option("-n", "--num-trials", default=1, type="int", help="the number of noise instantiations")
parser.add_option("-s", "--max-slides", default=np.infty, type="float", help="maximum number of slides per instantiation")

parser.add_option("", "--time", default=False, action="store_true")

parser.add_option("-t", "--tag", default="", type="string")
parser.add_option("-o", "--output-dir", default="./", type="string")

parser.add_option("", "--flush", default=False, action="store_true", help="write out pkl file after each noise instantiation")

opts, args = parser.parse_args()

#========================

opts.tau = np.array(sorted(list(set(opts.tau))))

if opts.tag:
	opts.tag = "_%s"%opts.tag

if opts.time:
	import time
	opts.verbose=True

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

### output filename
filename = "%s/data%s.pkl"%(opts.output_dir, opts.tag)

#=================================================
data = [] ### need relatively light storage structure
          ### for each noise realization, we store:
          ### 	seriesA (np.ndarray)
          ### 	seriesB (np.ndarray)
          ### 	seriesS (np.ndarray)
          ### 	for each tau, we store:
          ### 		num_coincs (zero-lag) (int)
          ###   	num_coincs_plus (all but zero-lag) (int)
          ###   	num_coincs_minus (all but zero-lag) (int)
          ### 		duration from timeslides (~number of slides) (float)

#========================
# iterate over noise instantiations
#========================
if opts.time:
	t_start = time.time()

for trialNo in xrange(opts.num_trials):
	datum = {}
	data.append( datum )

	if opts.verbose:
		print "===================================================="
		print "                  trial %d / %d"%(trialNo+1, opts.num_trials)
		print "===================================================="

	if opts.time:
		t_trial = time.time()

	### simulate series S
	if opts.verbose:
		print "simulating S with rate : %.3f"%(opts.rateS)

	if opts.time:
		t_S = time.time()

	S = su.poisson_series(opts.rateS, opts.dur, flag=True)
	num_S = len(S)
	exp_S = opts.rateS*opts.dur

	if opts.verbose:
		print "\texpected %10.3f +/- %.3f"%(exp_S, exp_S**0.5)
		print "\tobserved %10.3f"%(num_S)
		print "\t   z     %10.3f"%((num_S-exp_S)/exp_S**0.5)

	if opts.time:
		print "\t\tS COMPLETE", time.time() - t_S

	### simulate series A
	if opts.verbose:
		print "simulating A with rate : %.3f"%(opts.rateA)

	if opts.time:
		t_A = time.time()

	A = su.poisson_series(opts.rateA, opts.dur, flag=True)
	num_A = len(A)
	exp_A = opts.rateA*opts.dur

	if opts.verbose:
		print "\texpected %10.3f +/- %.3f"%(exp_A, exp_A**0.5)
		print "\tobserved %10.3f"%(num_A)
		print "\t   z     %10.3f"%((num_A-exp_A)/exp_A**0.5)

	if opts.time:
		print "\t\tA COMPLETE", time.time()-t_A

	### simulate series B
	if opts.verbose:
		print "simulating B with rate : %.3f"%(opts.rateB)

	if opts.time:
		t_B = time.time()

	B = su.poisson_series(opts.rateB, opts.dur, flag=True)
	num_B = len(B)
	exp_B = opts.rateB*opts.dur

	if opts.verbose:
		print "\texpected %10.3f +/- %.3f"%(exp_B, exp_B**0.5)
		print "\tobserved %10.3f"%(num_B)
		print "\t   z     %10.3f"%((num_B-exp_B)/exp_B**0.5)

	if opts.time:
		print "\t\tB COMPLETE", time.time() - t_B

	### document series
	datum.update( {"A":np.array(A, dtype=float), "B":np.array(B, dtype=float), "S":np.array(S, dtype=float)} )

	#================
	# iterate over coinc windows
	#================
	for tauNo, tau in enumerate(opts.tau):
		if opts.verbose:
			print "\t============================================"
			print "\t          tau %d / %d -> %.3f"%(tauNo+1, len(opts.tau), tau)
			print "\t============================================"

		if opts.time:
			t_tau = time.time()

		### add S to A, B and re-sort
		AS = A+S
		AS.sort(key=lambda l: l[0])

		BS = B+S
		BS.sort(key=lambda l: l[0])

		### perform coincidence
		if opts.time:
			t_coinc = time.time()

		coinc = su.all_coinc(AS, BS, tau, return_list=True)
		num_coinc = len(coinc)
		num_accid = num_coinc - num_S
		exp_coinc = (opts.rateS + 2*tau*(opts.rateA*opts.rateB + opts.rateS*(opts.rateA + opts.rateB)))*opts.dur
		exp_accid = 2*tau*(opts.rateA*opts.rateB + opts.rateS*(opts.rateA+opts.rateB))*opts.dur

		if opts.verbose:
			print "\tnumber of coinc in zero-lag:"
			print "\t\texpected %10.3f +/- %.3f"%(exp_coinc, exp_coinc**0.5)
			print "\t\tobserved %10.3f"%(num_coinc)
			print "\t\t    z    %10.3f"%((num_coinc-exp_coinc)/exp_coinc**0.5)

			print "\tknowing the number of signals:"
			print "\t\texpected %10.3f +/- %.3f"%(exp_accid, exp_accid**0.5)
			print "\t\tobserved %10.3f"%(num_accid)
			print "\t\t    z    %10.3f"%((num_accid - exp_accid)/exp_accid**0.5)

		if opts.time:
			print "\t\t\tCOINC COMPLETE:", time.time() - t_coinc

		### preform time-slides
		ASp = AS
		BSp = BS

		ASm = deepcopy(AS)
		BSm = deepcopy(BS)
		for tupA, tupB in coinc: ### remove zero-lag coincs
			try:
				ASm.remove(tupA)
			except ValueError:
#				print "tupA not found!"
				pass
			try:
				BSm.remove(tupB)
			except ValueError:
#				print "tupB not found!"
				pass

		num_slides = int(min(opts.max_slides, 0.5*opts.dur/(2*tau) - 1))
		if num_slides < 1:
			raise ValueError("num_slides < 1. That's bad...")
		dt = opts.dur/(num_slides + 1)
		if dt <= tau:
			raise ValueError("dt <= tau. That's bad...")

		if opts.verbose:
			print "\tprocessing %d slides with dt=%.5f"%(num_slides, dt)

		if opts.time:
			t_slides = time.time()

		num_Cp = 0
		num_Cm = 0
		slideDur = 0.0
		for slideNo in xrange(num_slides):
			### shift by one slide
			if opts.slide_type == "cyclic":
				BSp = su.cyclic_shift(BSp, dt, opts.dur, resort=True)
				BSm = su.cyclic_shift(BSm, dt, opts.dur, resort=True)
				slideDur += opts.dur
			elif opts.slide_type == "linear":
				BSp = su.linear_shift(BSp, dt, duration=opts.dur)
				BSm = su.linear_shift(BSm, dt, duration=opts.dur)
				slideDur += opts.dur - (slideNo+1)*dt
			else:
				raise ValueError("--slide-typ=%s not understood"%opts.shift_type)

			### count number of coincs
			num_Cp += su.all_coinc(ASp, BSp, tau, return_list=False)
			num_Cm += su.all_coinc(ASm, BSm, tau, return_list=False)

                exp_Cp = slideDur*2*tau*(opts.rateA+opts.rateS)*(opts.rateB+opts.rateS)
                exp_Cm = slideDur*2*tau*opts.rateA*opts.rateB*(1 - (1-np.exp(-2*tau*opts.rateA)) - (1-np.exp(-2*tau*opts.rateS)))*(1 - (1-np.exp(-2*tau*opts.rateB)) - (1-np.exp(-2*tau*opts.rateS)))

                if opts.verbose:
			print "\t\tslides with zero-lag coincs:"
                        print "\t\t\texpected %10.3f +/- %.3f"%(exp_Cp, exp_Cp**0.5)
                        print "\t\t\tobserved %10.3f"%(num_Cp)
                        print "\t\t\t    z    %10.3f"%((num_Cp-exp_Cp)/exp_Cp**0.5)

			print "\t\tslides without zero-lag coincs:"
                        print "\t\t\texpected %10.3f +/- %.3f"%(exp_Cm, exp_Cm**0.5)
                        print "\t\t\tobserved %10.3f"%(num_Cm)
                        print "\t\t\t    z    %10.3f"%((num_Cm-exp_Cm)/exp_Cm**0.5)

		if opts.time:
			print "\t\tSLIDES COMPLETE:", time.time() - t_slides
			print "\tCOINC-WINDOW COMPLETE:", time.time() - t_tau

		datum.update({tau:{"num_C":num_coinc, "num_Cp":num_Cp, "num_Cm":num_Cm, "slideDur":slideDur}})

	if opts.time:
		print "TRIAL COMPLETE:", time.time() - t_trial

	### save data
	if opts.flush:
		if opts.verbose:
			print "===================================================="
			print "  saving data to %s"%filename
			print "===================================================="

		if opts.time:
			t_save = time.time()

		file_obj = open(filename, "w")
		pickle.dump({"rateA":opts.rateA, "rateB":opts.rateB, "rateS":opts.rateS, "dur":opts.dur}, file_obj)
		pickle.dump(data, file_obj)
		file_obj.close()

		if opts.time:
			print "SAVE COMPLETE:", time.time()-t_save

#=================================================
### save data if we haven't already
if not opts.flush:
	if opts.verbose:
		print "===================================================="
		print "  saving data to %s"%filename
		print "===================================================="

	if opts.time:
		t_save = time.time()

	file_obj = open(filename, "w")
	pickle.dump({"rateA":opts.rateA, "rateB":opts.rateB, "rateS":opts.rateS, "dur":opts.dur}, file_obj)
	pickle.dump(data, file_obj)
	file_obj.close()

	if opts.time:
		print "SAVE COMPLETE:", time.time()-t_save

#=================================================
