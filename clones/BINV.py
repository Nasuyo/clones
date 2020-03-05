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
import struct
import numpy as np

def readBIN(filename,unpack=False):
    """Reads in a binary file written using the fortran RLFTlbx.
    Pretty slow currenlty so a cpp version is foreseen"""
    dictout={}
    #default to assuming the file is in little endian
    endianness='<'
    #open filename in binary mode
    with open(filename,'rb') as fid:
        #read the endianess checker
        (endian,)=struct.unpack(endianness+"H", fid.read(2))
        #compare the magic number (should be 18754 on a system with the same endiannes of the file)
        if endian != 18754:
            #switch to big endian
            endianness='>'

        dictout['version']='BI'+fid.read(6).decode('utf-8')
        vnum=float(dictout['version'][4:7])

        #read type, description etc

        dictout['type']=fid.read(8).decode('utf-8')
        dictout['description']=fid.read(80).decode('utf-8')

        #read integer meta information
        (nints,ndbls,nval1,nval2)=struct.unpack(endianness+'IIII',fid.read(4*4))

        if vnum < 2.4:
            (pval1,pval2)=struct.unpack(endianness+'II',fid.read(4*2))
        else:
            (pval1,pval2)=struct.unpack(endianness+'LL',fid.read(8*2))


        if vnum <= 2.1:
            if dictout["type"] in ['SYMV0___','BDFULLV0','BDSYMV0','BDFULLVN']:
                nvec=0
                pval2=1
            elif dictout["type"] == "SYMV1___":
                nvec=1
                pval2=1
            elif dictout["type"] == "SYMV2___":
                nvec=2
                pval2=1
            elif dictout["type"] == "FULLSQV0":
                nvec=0
                pval2=pval1

            nread=0
            nval2=nval1
        else:
            (nvec,nread)=struct.unpack(endianness+"II",fid.read(4*2))

        dictout["nval1"]=nval1
        dictout["nval2"]=nval2

        #read type dependent index data
        if dictout['type'] in ["BDSYMV0_","BDSYMVN_","BDFULLV0","BDFULLVN"]:
            (nblocks,)=struct.unpack(endianness+'I',fid.read(4))
            dictout["nblocks"]=nblocks

        if nread >0:
            dictout["readme"]=fid.read(nread*80).decode('utf-8')

        names=[]
        vals=[]
        if nints > 0:
            inames=np.fromfile(fid,dtype='|S24',count=nints).astype('|U24')
            names.extend(list(inames))
            if vnum <= 2.4:
                ivals=np.fromfile(fid,dtype=endianness+'I',count=nints)
            else:
                ivals=np.fromfile(fid,dtype=endianness+'L',count=nints)
            vals.extend(ivals)
        if ndbls >0:
            dnames=np.fromfile(fid,dtype='|S24',count=ndbls).astype('|U24')
            names.extend(list(dnames))
            dvals=np.fromfile(fid,dtype=endianness+'d',count=ndbls)
            vals.extend(dvals)

        if names:
            dictout["meta"]={ky.strip():val for ky,val in zip(names,vals)}

        #read side description data
        dictout["side1_d"]=np.fromfile(fid,dtype='|S24',count=nval1).astype('|U24')

        if dictout["type"] in ['BDSYMV0_','BDFULLV0','BDSYMVN_','BDFULLVN']:
            dictout["blockind"]=np.fromfile(fid,dtype=endianness+'I',count=nblocks)



        #possibly read second side description
        if dictout['type'] in ['BDFULLV0','BDFULLVN','FULLSQV0','FULLSQVN']:
            if vnum <= 2.2:
                dictout["side2_d"]=dictout["side1_d"]
            else:
                dictout["side2_d"]=np.fromfile(fid,dtype='|S24',count=nval1).astype('|U24')

        elif dictout["type"] == "FULL2DVN":
            dictout["side2_d"]=np.fromfile(fid,dtype='|S24',count=nval2).astype('|U24')


        # read vectors
        if nvec >0:
            dictout["vec"]=np.fromfile(fid,dtype=endianness+"d",count=nvec*nval1).reshape((nval1,nvec),'F')

        # read packed matrix
        pack=np.fromfile(fid,dtype=endianness+'d',count=pval1*pval2)


    if not unpack:
        dictout["pack"]=pack
    else:
        raise Exception("Unpacking not yet supported")






    return dictout
