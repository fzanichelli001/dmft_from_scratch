import numpy as np

def pade_analytic_continuation(wn, gf, norb = 1, nw = 500, nw_out = 1000, emin = -10.0, emax = 10.0, delta = 0.000000001):
	'''
	perform the analytic continuation form Matzubara frequencies to real frequencies of the green's function gf

	input:			wn[n]       imaginay part of the Matzubara frequencies n=0,...nw-1=500 matzubara frequencies
					gf[m,n]     green's function in Matzubara frequencies, m=0,...norb-1=1, n=0,...nw-1=500 matzubara frequencies
					norb		number of orbitals
					nw			number of matzubara frequencies
					nw_out		number of real output frequencies
					emin, emax	minimum and maximum of real frequency for the output GF
					delta       small parameter for the pade algorithm

	output:         gf_out[m,n]   green's function in real frequencies, m=0,...norb-1=1, n=0,...nw_out-1=1000 real frequencies
	'''

	de = (emax-emin)/nw_out

	# pade
	pmatrix = np.zeros((norb,nw,nw),dtype=complex)
	for m1 in range(norb):
		for n in range(nw):
			pmatrix[m1,0,n] = gf[m1,n]

		for m in range(1,nw):
			for n in range(1,m+1):
				if ( abs(pmatrix[m1,n-1,m]) < delta ):
					continue
				pmatrix[m1,n,m] = ( pmatrix[m1,n-1,n-1] - pmatrix[m1,n-1,m] ) / ( pmatrix[m1,n-1,m]*1.0j*( wn[m]-wn[n-1] ) )


	gf_out = np.zeros((norb,nw_out),dtype=complex)
	for nn in range(nw_out):
		# w  is the current real frequency
		w = emin + nn*de

		for m1 in range(norb):

			# find good n
			nstart = 0
			for m in range(nw):
				if ( abs(pmatrix[m1,m,m])<delta ):
					break
				nstart += 1
			nstart = min(nstart,nw-1)
			nstart = nw-1

			val = 1.0 + pmatrix[m1,nstart,nstart]*( w - 1.0j*wn[nstart-1] )
			for m in range(nstart-1,0,-1):
				val = 1.0 + pmatrix[m1,m,m]*( w - 1.0j*wn[m-1] )/val
			val = pmatrix[m1,0,0]/val

			# val is the current value of the green's function in real frequencies
			gf_out[m1, nn]=val
	
	return gf_out