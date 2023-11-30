import numpy as np

norb = 1
nw = 500
nw_out = 1000
emin = -10.0
emax = 10.0
inf = open("g_loc.dat",'r')

####################################
####################################
de = (emax-emin)/nw_out
delta = 0.000000001

gf = np.zeros((norb,nw),dtype=complex)
wn = np.zeros((nw))
for n in range(nw):
	data = [float(x) for x in inf.readline().split()]
	wn[n] = data[0]
	for m in range(norb):
		gf[m,n] = data[1 + 2*m] + data[1 + 2*m+1]*1.0j
inf.close()

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


outf = open('pade.dat','w')
for nn in range(nw_out):
	w = emin + nn*de
	outf.write(str(w) + '\t')

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


		outf.write(str(val.real) + '\t' + str(val.imag) + '\t')

	outf.write( '\n')
outf.close()

