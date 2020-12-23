import h5py
import glob
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
from datetime import datetime

from astropy.coordinates import EarthLocation
from astropy import units as u



def gather_ionograms(rmin=0,
                     rmax=750,
                     t0=1607080000.0,
                     t1=1607095800.0):

    fl0=glob.glob("oblique_data/lfm*.h5")
    fl0.sort()
    fl=[]
    
    for f in fl0:
        h=h5py.File(f,"r")
        if (h["t0"][()] > t0) and (h["t0"][()] < t1):
            fl.append(f)
        h.close()
        
    h=h5py.File(fl[0],"r")
    t0=h["t0"][()]
    dtau=t0-n.floor(t0)
    dr = dtau*c.c
    ranges=(n.copy(h["ranges"][()])+dr)/1e3
    ridx=n.where( (ranges > rmin) & (ranges < rmax) )[0]
    ranges=ranges[ridx]
    freqs=n.copy(h["freqs"][()])
    h.close()

    print(len(ranges))
    n_files=len(fl)
    n_ranges=len(ranges)
    n_freqs=len(freqs)
    print(n_freqs)
    print(n_ranges)
    print(n_files)
    S=n.zeros([n_files,n_ranges,n_freqs],dtype=n.float16)
    tv=[]

    for fi,f in enumerate(fl):
        print(f)
        h=h5py.File(f,"r")
        tnow = h["t0"][()]
        tv.append(tnow)
        S0=n.transpose(h["S"][()])
        print(S0.shape)
        for fri in range(S0.shape[1]):
            noise_floor=n.median(S0[:,fri])
            S0[:,fri]=(S0[:,fri]-noise_floor)/noise_floor
            
        S0[S0<=0]=1e-3
        S0[n.isnan(S0)]=1e-3
        S[fi,:,:]=S0[ridx,:]
        
        plt.pcolormesh(freqs/1e6,ranges,10.0*n.log10(S0[ridx,:]),vmin=0,vmax=20.0)
        plt.title(datetime.utcfromtimestamp(tnow).strftime('%Y-%m-%d %H:%M:%S'))

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Virtual range (km)")
        plt.xlim([0.5,16])
        plt.tight_layout()
        plt.savefig("figs/ski-%06d.png"%(tnow))
        plt.clf()
        plt.close()
        h.close()
    ho=h5py.File("oblique_data/sod_to_ski.h5","w")
    ho["unix_time"]=tv
    ho["SNR"]=S
    ho["ranges"]=ranges
    ho["freqs"]=freqs/1e6
    ho.close()

if __name__ == "__main__":
    gather_ionograms(rmin=400,
                     rmax=550,
                     t0=1607084000.0,
                     t1=1607095800.0)

