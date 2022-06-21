import matplotlib

SMALL_SIZE = 14
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

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


def sod_filter(f):
    """
    four sample coherent integration with 13 microsecond sample spacing
    """
    T_s = 1.0/2144.0
    omhat = 2.0*n.pi*f*T_s
    hmag=n.zeros(len(f),dtype=n.complex64)
    for i in range(4):
        hmag+=n.exp(-1j*omhat*float(i))
    hmag=n.abs(hmag)**2.0
    hmag=hmag/16.0
    return(hmag)

    
def rcs_est(G=2.0,
            P_tx=15e3,
            lam=c.c/36.9e6,
            T_sys=9000,
            B=50e3,
            R=170e3,
            SNR=100.0):
    
    return(SNR*c.k*T_sys*B*(4.0*n.pi)**3.0*R**4/(P_tx*G**2.0*lam**2.0))

def range_to_height(radar_pos,max_t=41.0,frad=36.9e6):
    """
    Use optical trajectory to map radar range to height
    """
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
    ts=[]
    for pi,pos in enumerate(p):
        if t[pi] < t0+max_t:
            az,el,r=jcoord.geodetic_to_az_el_r(radar_pos[0], radar_pos[1], 0.0, lat[pi], lon[pi], hg[pi]*1e3)
            print((n.linalg.norm(radar_p-pos)/1e3))
            print("%1.2f %1.2f %1.2f"%(az,el,r/1e3))
            ranges.append((n.linalg.norm(radar_p-pos)/1e3))
            heights.append(hg[pi])
            els.append(el)
            ts.append(t[pi])
    ts=n.array(ts)
    ranges=n.array(ranges)
    heights=n.array(heights)
#    gidx=n.where(heights > 90)[0]
#    pf=n.polyfit(ranges[gidx],heights[gidx],1)
 #   hfun=n.poly1d(pf)    
    hfun=sint.interp1d(ranges,heights)    

    dopvels=n.diff(ranges)/n.diff(ts)
    tdop=0.5*(ts[0:(len(ts)-1)]+ts[1:len(ts)])
    rdop=0.5*(ranges[0:(len(ranges)-1)]+ranges[1:len(ranges)])    
    dopfun = sint.interp1d(rdop,dopvels)
    dopfreqfun=sint.interp1d(hfun(rdop),2.0*frad*dopvels*1e3/c.c)
    if False:
        plt.plot(tdop,dopvels,".")
        plt.show()
        
        plt.plot(tdop,2.0*frad*dopvels*1e3/c.c,".")
        plt.show()
        
    if False:
#        rgs=n.linspace(163,220,num=1000)        
        plt.plot(ranges,heights,"x")
        plt.plot(ranges,hfun(ranges))    
        plt.show()
    
#    ranges[0]=300.0
 #   ranges[-1]=150
    
    elfun=sint.interp1d(ranges,els)

    if True:
        plt.plot(ranges,els,"x")
#        rgs=n.linspace(150,300,num=1000)
        plt.plot(ranges,elfun(ranges))    
        plt.show()
        
    return(hfun,elfun,dopfun,dopfreqfun,tfun)


def rcs_sod(P_tx=30*2*15e3, f=36.9e6, T_sys=9000, B=77e3, K=10.5):
    radar_pos=[sc.sod_coords[0],sc.sod_coords[1]]
    hfun,elfun,dopfun,dopfreq,tfun=range_to_height(radar_pos)
    h=h5py.File("rcs/Sodankylä_head.h5","r")
    rg=h["r"][()]
    snr=10**(h["snr"][()]/10.0)
    t=h["t"][()]
    plt.plot(t,elfun(rg))
    plt.show()

    n_snr=len(t)
    rcs_h=[]
    rcs=[]
    rcs_n=[]    
    rcs_p=[]
    rcs_m=[]
    filter_gains=[]
    for i in range(n_snr):
        hgt=hfun(rg[i])
        el=elfun(rg[i])
        gain=10**(gp.skymet_gain(el)/10.0)
        
        filter_gain=sod_filter(n.array([dopfreq(hgt)]))[0]
        print("fg %1.2f df %1.2f"%(filter_gain,dopfreq(hgt)))
        signal=snr[i]
        signal=signal/filter_gain

        if snr[i] > 1.0 and filter_gain > 0.001:
            
            noise=1.0
            snr_std = (signal+1.0)/n.sqrt(K)

            filter_gains.append(filter_gain)
            
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

    return(rcs,rcs_n,rcs_h,xerr,filter_gains,dopfreq)


def rcs_and(P_tx=30*4*15e3, f=32.9e6, T_sys=9000, B=77e3, K=10.5):
    
    radar_pos=[sc.and_coords[0],sc.and_coords[1]]
    hfun,elfun,dopfun,dopfreq,tfun=range_to_height(radar_pos)
    h=h5py.File("rcs/Andenes_head.h5","r")
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

    return(rcs,rcs_n,rcs_h,xerr)

def plot_rcs():
    s_rcs, s_rcs_n, s_rcs_h, s_xerr, filter_gains, dopfreq = rcs_sod(P_tx=30*4.0*15e3, f=36.9e6, T_sys=9000, B=77e3, K=7.0)
    a_rcs, a_rcs_n, a_rcs_h, a_xerr = rcs_and(P_tx=7*125*30e3, f=32.55e6, T_sys=9000, B=100e3, K=5.0)


    ho=h5py.File("rcs/rcs_est.h5","w")
    ho["s_rcs"]=s_rcs
    ho["s_rcs_h"]=s_rcs_h
    ho["s_rcs_e"]=s_xerr
    ho["a_rcs"]=a_rcs
    ho["a_rcs_h"]=a_rcs_h
    ho["a_rcs_e"]=a_xerr
    ho.close()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.errorbar(s_rcs,s_rcs_h,xerr=s_xerr,fmt="o",label="Sodankylä",color="C0")
    hgt=n.linspace(82,107.5,num=100)    
    ax2.plot(sod_filter(dopfreq(hgt)),hgt,color="gray",label="Sodankylä magnitude response")
#    plt.plot(s_rcs_n, s_rcs_h,color="C0")
    ax1.errorbar(a_rcs,a_rcs_h,xerr=a_xerr,fmt="o",color="C1",label="Andenes")
#    plt.plot(a_rcs_n, a_rcs_h,color="C1")
    ax1.set_xlabel("Radar cross-section (m$^2$)")
    ax1.set_ylabel("Height (km)")
    ax2.set_xlabel("Coherent integration magnitude response")
    ax1.set_xlim([0,30e3])
    ax1.legend()
    ax2.legend(loc=4)    
    plt.tight_layout()
    plt.savefig("figs/rcs_est.png")
    plt.show()
            


def plot_sod_resp():
#            filter_gain=sod_filter(n.array([dopfreq(rg[i])]))[0]
 #       print("fg %1.2f df %1.2f"%(filter_gain,dopfreq(rg[i])))

    f=n.linspace(-6e3,0,num=1000)
    plt.plot(f,sod_filter(f))
    plt.xlabel("Doppler frequency (Hz)")
    plt.ylabel("Normalized coherent integration gain (linear)")
    plt.title("Four pulse coherent integration filter")
    plt.savefig("figs/cohint_magresp.png")
    plt.show()

if __name__ == "__main__":
    plot_sod_resp()
    print(plot_rcs())
    
    
            
