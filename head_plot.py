import h5py
import glob
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
from datetime import datetime
import stuffr

from astropy.coordinates import EarthLocation
from astropy import units as u


def geometry_test(pos0=[67.365524, 26.637495],
                  pos1=[69.3481870270324, 20.36342676479248],
                  title="None",
                  one_way=False,
                  radar_dt=0.0,
                  rlim=[None,None],
                  tlim=[None,None],
                  vmax=40,
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

    t13=stuffr.date2unix(2020,12,4,13,30,0)
    print(t13)
    dB=n.transpose(10.0*n.log10(hh["RTI"][()]))
    dB=dB-n.nanmedian(dB)

    
#    plt.pcolormesh(hh["tvec"][()]-radar_dt-t13,hh["ranges"][()],dB,vmin=-2,vmax=40)
    #hh["tvec"][()]-radar_dt-t13,hh["ranges"][()]
    t_ext=hh["tvec"][()]-radar_dt-t13
    dt_im=n.diff(t_ext)[0]
    r_ext=hh["ranges"][()]
    dr_im=n.diff(r_ext)[0]

    fig,ax=plt.subplots()
    im=ax.imshow(dB,extent=[n.min(t_ext),n.max(t_ext)+dt_im,n.min(r_ext),n.max(r_ext)+dr_im],aspect="auto",origin="lower",vmin=0,vmax=vmax)


    
    plt.title(title)
    ax.plot(tt-t13,ranges,color="white",alpha=0.5)
    if tlim[0] != None:
        plt.xlim(tlim)
    if rlim[0] != None:
        plt.ylim(rlim)    
    plt.xlabel("Time (seconds since 13:30 UTC)")
    plt.ylabel("One-way range (km)")
    cb=plt.colorbar(im)
    cb.set_label("SNR (dB)")

    head_r=[]
    head_t=[]
    head_snr=[]
    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.button == 3:
            xi=int(n.floor((event.xdata-n.min(t_ext))/dt_im))
            yi=int(n.floor((event.ydata-n.min(r_ext))/dr_im))        
            head_t.append(event.xdata)
            head_r.append(event.ydata)
            head_snr.append(dB[yi,xi])
            print("%1.2f %1.2f=%1.2f"%(event.xdata,event.ydata,dB[yi,xi]))
            ho=h5py.File("%s_head.h5"%(title),"w")
            ho["t"]=head_t
            ho["r"]=head_r
            ho["snr"]=head_snr
            ho.close()
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    
    plt.tight_layout()
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
                  tlim=[34,47],
                  vmax=20,
                  one_way=True,
                  title="Andenes")


    geometry_test(pos0=sod_coords,pos1=sod_coords,pfname="figs/sod_sod_rti_head.png",
                  one_way=True,
                  radar_dt=0.6,
                  rlim=[152,202],
                  tlim=[34,47],
                  vmax=40,
                  title="Sodankylä")

