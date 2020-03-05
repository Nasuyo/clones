# This file is part of Frommle
# Frommle is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# Frommle is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with Frommle; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

# Author Roelof Rietbroek (roelof@geod.uni-bonn.de), 2018
# Change by Stefan Schröder (s7stschr@uni-bonn.de), 2019: use pyshtools
# instead of frommles shdata

from clones.BINV import readBIN
from clones import harmony
import pyshtools as sh
import numpy as np
import math

class DDKfilter():
    def __init__(self,filtername,transpose=False):
        """Reads filter coefficients from file.
        
        Transpose causes the filter to be applied in its transpose form (e.g.
        needed for filtering basin before basin averaging operations).
        """

        W=readBIN(filtername)
        if W['type'] != "BDFULLV0":
            raise Exception("Not an appropriate filter matrix")
        self.nmax=int(W['meta']['Lmax'])
        self.blocks=[]
        self.mtindx=[]

        lastblckind=0
        lastindex=0
        #unpack matrix in diagonal blocks
        for iblck in range(W['nblocks']):
            #order of the block
            m=int((iblck+1)/2)
            # get the trigonometic part of the block (cosine=0, sin=1)
            if iblck == 0:
                trig=0
            else:
                (dum,trig)=divmod(iblck-1,2)
        #size of the block
            sz=W['blockind'][iblck]-lastblckind

            self.mtindx.append((m,trig))
            self.blocks.append(np.identity(W['meta']["Lmax"]+1-m))

            nminblk=max(int(W["meta"]["Lmin"]),m)

            shft=nminblk-m

            #note the blocks are in fact stored as Fortran style column major order
            #but we load them here pretending them to be row major so that the np.dot operation acts from __call()
            #acts correctly unless transpose is set to true
            if transpose:
                transp='F'
            else:
                transp='C'

            self.blocks[-1][shft:,shft:]=W['pack'][lastindex:lastindex+sz*sz].reshape([sz,sz],order=transp)

            lastblckind=W["blockind"][iblck]
            lastindex+=sz*sz

    def __call__(self,incoef):
        """Filter the input coefficients.
        
        Eats a pyshtools.SHCoeffs and drops out one as well.
        """
        # TODO: Params

        if incoef.lmax > self.nmax:
            print(incoef.lmax, self.nmax)
            raise ValueError("Maximum degree of filter matrix is smaller than the maximum input degree")
#        shfilt=shdata(incoef.nmax)
        shfilt = sh.SHCoeffs.from_array(np.zeros((2, incoef.lmax+1,
                                                 incoef.lmax+1)))
        # coefficients as vectors
        C = harmony.sh_mat2vec(incoef.coeffs[0])
        S = harmony.sh_mat2vec(incoef.coeffs[1])
        C_out = harmony.sh_mat2vec(shfilt.coeffs[0])
        S_out = harmony.sh_mat2vec(shfilt.coeffs[1])
        #loop over the filter blocks
        for (m,trig),block in zip(self.mtindx,self.blocks):
            if m> incoef.lmax:
                break
#            st=shfilt.idx(m,m)
#            nd=shfilt.idx(incoef.nmax,m)+1
            st = harmony.sh_nm2i(m, m, shfilt.lmax)
            nd = harmony.sh_nm2i(incoef.lmax, m, shfilt.lmax)+1
            ndblck=incoef.lmax-m+1
            if trig == 0:
#                np.dot(incoef.C[st:nd],block[:ndblck,:ndblck],out=shfilt.C[st:nd])
                np.dot(C[st:nd], block[:ndblck,:ndblck], out=C_out[st:nd])
            else:
#                np.dot(incoef.S[st:nd],block[:ndblck,:ndblck],out=shfilt.S[st:nd])
                np.dot(S[st:nd], block[:ndblck,:ndblck], out=S_out[st:nd])
        CS = np.array([harmony.sh_vec2mat(C_out, shfilt.lmax),
                       harmony.sh_vec2mat(S_out, shfilt.lmax)])
        shfilt.coeffs = CS
        return shfilt
