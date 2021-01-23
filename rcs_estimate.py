import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c

import stuffr
import station_coords as sc
import h5py
import scipy.interpolate as sint
import jcoord

from astropy.coordinates import EarthLocation
from astropy import units as u
import gain_pattern as gp

def rcs_est(G=2.0,
            P_tx=30*4*15e3,
            lam=c.c/36.9e6,
            T_sys=9000,
            B=50e3,
            R=170e3,
            SNR=100.0):
    
    return(SNR*c.k*T_sys*B*(4.0*n.pi)**3.0*R**4/(P_tx*G**2.0*lam**2.0))

def range_to_height(radar_pos,max_t=41.0):
    t0=stuffr.date2unix(2020,12,4,13,30,0)
    
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    p=ht["ecef_pos_m"][()]
    hg=ht["h_km_wgs84"][()]
    t=ht["t_unix"][()]
    lat=ht["lat_wgs84"][()]
    lon=ht["lon_wgs84"][()]

    pf=n.polyfit(p[:,0],p[:,1],1)
    yfun=n.poly1d(pf)    
    pf=n.polyfit(p[:,0],p[:,2],1)
    zfun=n.poly1d(pf)    
    pf=n.polyfit(p[:,0],hg,1)
    hfun=n.poly1d(pf)    
    pf=n.polyfit(p[:,0],lat,1)
    latfun=n.poly1d(pf)    
    pf=n.polyfit(p[:,0],lon,1)
    lonfun=n.poly1d(pf)
    
    pf=n.polyfit(p[:,0],t,1)
    tfun=n.poly1d(pf)

    

    x=n.linspace(n.min(p[:,0])-100e3,n.max(p[:,0])+100e3,num=500)
    
    if False:
        plt.plot(p[:,0],t,".")
        plt.plot(x,tfun(x))    
        plt.show()
        plt.plot(p[:,0],p[:,1],".")
        plt.plot(x,yfun(x))    
        plt.show()
        plt.plot(p[:,0],p[:,2],".")
        plt.plot(x,zfun(x))        
        plt.show()
        plt.plot(p[:,0],hg,".")
        plt.plot(x,hfun(x))        
        plt.show()
        plt.plot(p[:,0],lat,".")
        plt.plot(x,latfun(x))            
        plt.show()
        plt.plot(p[:,0],lon,".")
        plt.plot(x,lonfun(x))                
        plt.show()
    
    #print(p.shape)
    #print(ht.keys())
    #exit(0)
    hg=hfun(x)
    p=n.zeros([len(x),3])
    p[:,0]=x
    p[:,1]=yfun(x)
    p[:,2]=zfun(x)
    lat=latfun(x)
    lon=lonfun(x)
    t=tfun(x)        
    loc = EarthLocation(lat=radar_pos[0]*u.deg,
                            lon=radar_pos[1]*u.deg,
                            height=0.0)
    
    radar_p=n.array(loc.itrs.cartesian.xyz)

    ranges=[]
    heights=[]
    els=[]        
    for pi,pos in enumerate(p):
        if t[pi] < t0+max_t:
            az,el,r=jcoord.geodetic_to_az_el_r(radar_pos[0], radar_pos[1], 0.0, lat[pi], lon[pi], hg[pi]*1e3)
            print((n.linalg.norm(radar_p-pos)/1e3))
            print("%1.2f %1.2f %1.2f"%(az,el,r/1e3))
            ranges.append((n.linalg.norm(radar_p-pos)/1e3))
            heights.append(hg[pi])
            els.append(el)

    ranges=n.array(ranges)
    heights=n.array(heights)
    gidx=n.where(heights > 90)[0]
#    pf=n.polyfit(ranges[gidx],heights[gidx],1)
 #   hfun=n.poly1d(pf)    
    hfun=sint.interp1d(ranges,heights)    

    if False:
        rgs=n.linspace(163,220,num=1000)        
        plt.plot(ranges,heights,"x")
        plt.plot(rgs,hfun(rgs))    
        plt.show()
    
    ranges[0]=300.0
    ranges[-1]=150
    
    elfun=sint.interp1d(ranges,els)

    if False:
        plt.plot(ranges,els,"x")
        rgs=n.linspace(150,300,num=1000)
        plt.plot(rgs,elfun(rgs))    
        plt.show()
        
    return(hfun,elfun)


def rcs_sod(P_tx=30*4*15e3, f=36.9e6, T_sys=9000, B=77e3, K=10.5):
    radar_pos=[sc.sod_coords[0],sc.sod_coords[1]]
    hfun,elfun=range_to_height(radar_pos)
    h=h5py.File("rcs/Sodankylä_head.h5","r")
    rg=h["r"][()]
    snr=10**(h["snr"][()]/10.0)
    t=h["t"][()]

    n_snr=len(t)
    rcs_h=[]
    rcs=[]
    rcs_n=[]    
    rcs_p=[]
    rcs_m=[]    
    for i in range(n_snr):
        hgt=hfun(rg[i])
        el=elfun(rg[i])
        gain=10**(gp.skymet_gain(el)/10.0)

        if snr[i] > 1.0:
            
            noise=1.0
            signal=snr[i]
            snr_std = (signal+1.0)/n.sqrt(K)
            
            rcsn0=rcs_est(G=gain,
                          P_tx=P_tx,
                          lam=c.c/f,
                          T_sys=T_sys,
                          B=B,
                          R=rg[i]*1e3,
                          SNR=1.0)
            rcs_n.append(rcsn0)
           
            
            rcs0=rcs_est(G=gain,
                        P_tx=P_tx,
                        lam=c.c/f,
                        T_sys=T_sys,
                        B=B,
                        R=rg[i]*1e3,
                        SNR=snr[i])
            rcs0_p=rcs_est(G=gain,
                          P_tx=P_tx,
                          lam=c.c/f,
                          T_sys=T_sys,
                          B=B,
                          R=rg[i]*1e3,
                          SNR=snr[i]+snr_std)
            rcs0_m=rcs_est(G=gain,
                          P_tx=P_tx,
                          lam=c.c/f,
                          T_sys=T_sys,
                          B=B,
                          R=rg[i]*1e3,
                          SNR=n.max([0,snr[i]-snr_std]))
            
            rcs.append(rcs0)
            rcs_p.append(rcs0_p)
            rcs_m.append(rcs0_m)            
            rcs_h.append(hgt)
            print("R=%1.2f snr=%1.2f+/-%1.2f el=%1.2f G=%1.2f rcs=%1.2f<%1.2f<%1.2f"%(rg[i],snr[i],snr_std,el,gp.skymet_gain(el),rcs0_m,rcs0,rcs0_p))
    xerr=n.zeros([2,len(rcs)])
    rcs=n.array(rcs)
    rcs_p=n.array(rcs_p)
    rcs_m=n.array(rcs_m)        
    xerr[0,:]=n.abs(rcs_m-rcs)
    xerr[1,:]=n.abs(rcs_p-rcs)
    plt.errorbar(rcs,rcs_h,xerr=xerr,fmt="o")
    plt.plot(rcs_n,rcs_h,color="gray")
 #   plt.plot(rcs_m,rcs_h)
#    plt.plot(rcs_p,rcs_h)    
    plt.xlabel("Radar cross-section (m$^2$)")
    plt.ylabel("Height (km)")
    plt.show()
            

    h.close()


if __name__ == "__main__":
    print(rcs_sod())
    
    
            
