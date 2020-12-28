import h5py
import glob
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
from datetime import datetime

from astropy.coordinates import EarthLocation
from astropy import units as u

def geometry_test(pos0=[67.365524, 26.637495],
                  pos1=[69.3481870270324, 20.36342676479248],
                  title="None",
                  one_way=False,
                  radar_dt=0.0,
                  rlim=[None,None],
                  tlim=[None,None],                  
                  rtname="sod_head/out-0.06s.h5",
                  pfname="sod_ski_aspec.png"):

    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    tt=ht["t_unix"][()]
    
    p=ht["ecef_pos_m"][()]
    hg=ht["h_km_wgs84"][()]
    print(p.shape)
    print(ht.keys())
    hh=h5py.File(rtname,"r")
    
    sod_loc = EarthLocation(lat=pos0[0]*u.deg,
                            lon=pos0[1]*u.deg,
                            height=400.0)
    ski_loc = EarthLocation(lat=pos1[0]*u.deg,
                            lon=pos1[1]*u.deg,
                            height=200.0)

    sod_p=n.array(sod_loc.itrs.cartesian.xyz)
    ski_p=n.array(ski_loc.itrs.cartesian.xyz)    

    ranges=[]
    for pi,pos in enumerate(p):
        ranges.append((n.linalg.norm(sod_p-pos)+n.linalg.norm(ski_p-pos))/1e3/2.0)

    plt.pcolormesh(hh["tvec"][()]-radar_dt,hh["ranges"][()],n.transpose(10.0*n.log10(hh["RTI"][()])))
    plt.plot(tt,ranges,color="white",alpha=0.5)
    if tlim[0] != None:
        plt.xlim(tlim)
    if rlim[0] != None:
        plt.ylim(rlim)    
    plt.xlabel("Time (unix)")
    plt.ylabel("One-way range (km)")
    plt.colorbar()
    plt.savefig(pfname)
    plt.show()
        

if __name__ == "__main__":

    sod_coords=[67.36558301653871, 26.638078807476912]
    and_coords=[69.29807478964716,16.043168307014426]    
    #    ski_coords=[69.3481870270324, 20.36342676479248] 

    geometry_test(pos0=and_coords,
                  pos1=and_coords,
                  pfname="figs/and_and_rti_head.png",
                  rtname="and_head/and-trail-0.2s.h5",
                  rlim=[335,415],
                  tlim=[34+1.6070886e9,47+1.6070886e9],
                  one_way=True,
                  title="Andenes-Andenes")


    geometry_test(pos0=sod_coords,pos1=sod_coords,pfname="figs/sod_sod_rti_head.png",
                  one_way=True,
                  radar_dt=0.6,
                  rlim=[152,202],
                  tlim=[34+1.6070886e9,47+1.6070886e9],                  
                  title="Sodankylä-Sodankylä")

