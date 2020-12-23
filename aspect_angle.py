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
                  pfname="sod_ski_aspec.png"):

    ht=h5py.File("camera_data/2020-12-04-trajectory.h5","r")
    p=ht["ecec_pos_m"][()]
    hg=ht["h_km_wgs84"][()]
    print(p.shape)
    print(ht.keys())

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
        if one_way:
            ranges.append((n.linalg.norm(sod_p-pos)+n.linalg.norm(ski_p-pos))/1e3/2.0)
        else:
            ranges.append((n.linalg.norm(sod_p-pos)+n.linalg.norm(ski_p-pos))/1e3)

    line=p[0,:]-p[-1,:]
    line=line/n.linalg.norm(line)
    thetas=[]
    for pi,pos in enumerate(p):
        k_i=(sod_p-pos)
        k_s=(ski_p-pos)
        k_i=k_i/n.linalg.norm(k_i)
        k_s=k_s/n.linalg.norm(k_s)
        k_b=k_i+k_s
        k_b=k_b/n.linalg.norm(k_b)
        thetas.append(n.arccos(n.dot(line,k_b)))
    thetas=180.0*n.array(thetas)/n.pi
    mi=n.argmin(n.abs(thetas-90.0))
    


    plt.scatter(ranges,hg,c=thetas)
    plt.plot([ranges[mi]],[hg[mi]],"x",color="red")    
    plt.title(title)
    print(mi)
    cb=plt.colorbar()
    cb.set_label("Aspect angle (deg")
    if one_way:
        plt.xlabel("One-way range (km)")
    else:
        plt.xlabel("Range (km)")
    plt.ylabel("Trail height (km)")
    plt.savefig(pfname)
    plt.show()
        

if __name__ == "__main__":
    sod_coords=[67.365524, 26.637495]
    ski_coords=[69.3481870270324, 20.36342676479248]

    geometry_test(pos0=sod_coords,pos1=ski_coords,pfname="figs/sod_ski_aspect.png",
                  title="Sodankylä-Skibotn")

    geometry_test(pos0=sod_coords,pos1=sod_coords,pfname="figs/sod_sod_aspect.png",
                  one_way=True,
                  title="Sodankylä-Sodankylä")

