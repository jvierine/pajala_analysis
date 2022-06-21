import h5py
import numpy as n
import matplotlib.pyplot as plt
import scipy.optimize as so
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import sys
import station_coords as stc
import jcoord
import scipy.interpolate as si
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import ITRS, SkyCoord

def fit_spec(spec,vel,v0=0,s0=50.0,plot=False):
    def ss(x):
        vel0=x[0]
        vels=x[1]
        model=n.exp(-(vel-vel0)**2.0/(2*vels**2.0))
        model=model/n.max(model)
        return(n.sum(n.abs(model-spec)**2.0))
    xhat=so.fmin(ss,[v0,s0])

    if plot:
        model=n.exp(-(vel-xhat[0])**2.0/(2*xhat[1]**2.0))
        model=model/n.max(model)
        
        plt.plot(vel,spec,".")    
        plt.plot(vel,model)
        plt.show()
    return(xhat)

def and_prof():
    and_files=["/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088637.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088639.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088641.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088643.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088645.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088647.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088649.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088651.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088653.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/snr/snr-1607088655.h5"]

    h=h5py.File(and_files[0],"r")
    ranges=n.copy(h["ranges"][()])
    vels=n.copy(h["vels"][()])
    rds=n.copy(h["S"][()])
    rds[:,:]=0.0
    h.close()
    for f in and_files:
        h=h5py.File(f,"r")
        rds+=h["S"][()]
        h.close()
    rds[rds<=0]=1e-3

    dopv=[]
    rangev=[]
    def press(event):
#        global dopv,rangev
        x, y = event.xdata, event.ydata
        dopv.append(x)
        rangev.append(y)
        ho=h5py.File("and_rv.h5","w")
        ho["dop"]=dopv
        ho["range"]=rangev
        ho.close()
        print("press %f %f"%(x,y))
        sys.stdout.flush()
        if event.key == '1':
            ax.plot(x,y,"x",color="red")
            fig.canvas.draw()
            
    fig, ax = plt.subplots(figsize=(14,10))
    fig.canvas.mpl_connect('key_press_event', press)

    
    im=ax.imshow(10.0*n.log10(rds),vmin=0,vmax=40,aspect="auto",extent=[n.min(vels),n.max(vels),n.min(ranges),n.max(ranges)],origin="lower")
    cb=plt.colorbar(im)
    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("One-way range (km)")
    cb.set_label("SNR (dB)")
    plt.xlim([-120,120])
    plt.ylim([340,410])
    
    plt.show()

def and_range_to_height(r0,plot=False):
    and_p=jcoord.geodetic2ecef(stc.and_coords[0], stc.and_coords[1], stc.and_coords[2])
    
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    p=ht["ecef_pos_m"][()]
    hg=ht["h_km_wgs84"][()]
    n_p=p.shape[0]
    ranges=[]
    heights=[]    
    for i in range(n_p):
        ranges.append(n.linalg.norm(and_p-p[i,:])/1e3)
        heights.append(hg[i])
    ranges=n.array(ranges)
    heights=n.array(heights)    
    pp=n.polyfit(ranges,heights,5)
    pfr=n.poly1d(pp)
    if plot:
        ranges0=n.linspace(n.min(ranges)-50,n.max(ranges)+50,num=200)
        plt.plot(ranges,heights,".")    
        plt.plot(ranges0,pfr(ranges0))
        plt.show()
        print(p.shape)
    return(pfr(r0))


def height_to_k(h,r_p,plot=False):
    """
    k vector in local horizontal coordinate system
    """
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    p=ht["ecef_pos_m"][()]
    hg=ht["h_km_wgs84"][()]
    n_p=p.shape[0]
    px=[]
    py=[]
    pz=[]        
    heights=[]
    
    for i in range(n_p):
        px.append(p[i,0])
        py.append(p[i,1])
        pz.append(p[i,2])        
        heights.append(hg[i])
        
    px=n.array(px)
    py=n.array(py)
    pz=n.array(pz)    
    heights=n.array(heights)
    
    ppx=n.polyfit(heights,px,5)
    ppy=n.polyfit(heights,py,5)
    ppz=n.polyfit(heights,pz,5)
    
    pfx=n.poly1d(ppx)
    pfy=n.poly1d(ppy)
    pfz=n.poly1d(ppz)

    if plot:
        hgs0=n.linspace(75,103,num=200)
        plt.plot(hg,px-pfx(hg),".")    
        plt.plot(hg,py-pfy(hg),".")
        plt.plot(hg,pz-pfz(hg),".")        
        plt.show()
        print(p.shape)

    k0=n.array([r_p[0]-pfx(h),r_p[1]-pfy(h),r_p[2]-pfz(h)])
    k0=k0/n.linalg.norm(k0)
    
    # convert to local observer centric
    llh=jcoord.ecef2geodetic(pfx(h),pfy(h),pfz(h))
    north=jcoord.enu2ecef(llh[0],llh[1],llh[2],0.0,1.0,0.0)
    east=jcoord.enu2ecef(llh[0],llh[1],llh[2],1.0,0.0,0.0)
    return(n.array([-1.0*n.dot(east,k0),-1.0*n.dot(north,k0)]))


def plot_and_prof():
    h=h5py.File("and_head/and_rv.h5","r")
    a_dops=n.copy(h["dop"][()])
    a_ranges=n.copy(h["range"][()])
    a_hgts=and_range_to_height(a_ranges)
    h.close()
    h=h5py.File("sod_data/trail_range_dop.h5","r")
    s_dops=n.copy(h["vel"][()])
    s_hgts=n.copy(h["h_km"][()])
    s_snr=n.copy(h["snr"][()])
    h.close()
    plt.subplot(121)
    plt.plot(a_dops,a_hgts,".")
    plt.title("Andenes")
    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("Height (km)")
    plt.ylim([75,105])
    plt.xlim([-75,75])    
    plt.subplot(122)
    plt.plot(s_dops,s_hgts,".")
    plt.title("Sodankylä")
    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("Height (km)")    
    plt.ylim([75,105])
    plt.xlim([-75,75])
    plt.tight_layout()
    plt.savefig("figs/and_sod_radial_vel_prof.png")
    plt.show()


def invert_wind(L=1.5,hg=n.arange(75,103,0.5)):
    """
    estimate zonal and meridional wind
    """
    h=h5py.File("and_head/and_rv.h5","r")
    a_dops=n.copy(h["dop"][()])
    a_ranges=n.copy(h["range"][()])
    a_hgts=and_range_to_height(a_ranges)
    h.close()
    h=h5py.File("sod_data/trail_range_dop.h5","r")
    s_dops=n.copy(h["vel"][()])
    s_hgts=n.copy(h["h_km"][()])
    s_snr=n.copy(h["snr"][()])
    h.close()
    
    and_p=jcoord.geodetic2ecef(stc.and_coords[0], stc.and_coords[1], stc.and_coords[2])
    sod_p=jcoord.geodetic2ecef(stc.sod_coords[0], stc.sod_coords[1], stc.sod_coords[2])

    Z=n.zeros(len(hg))
    Z_std=n.zeros(len(hg))    
    M=n.zeros(len(hg))
    M_std=n.zeros(len(hg))    
    for hi,h in enumerate(hg):
        a_gidx=n.where(n.abs(a_hgts-h)<L)[0]
        s_gidx=n.where(n.abs(s_hgts-h)<L)[0]
        n_m=len(a_gidx)+len(s_gidx)
        A=n.zeros([n_m,2])
        m=n.zeros(n_m)
        mi=0
#        plt.plot(s_dops[s_gidx],s_hgts[s_gidx],".")
 #       plt.show()
        for i in a_gidx:
            and_k=height_to_k(a_hgts[i],and_p)
            A[mi,:]=and_k
            m[mi]=a_dops[i]
            mi+=1
        
        for i in s_gidx:
            sod_k=height_to_k(s_hgts[i],sod_p)
            A[mi,:]=sod_k
            m[mi]=s_dops[i]            
            mi+=1
        std_est0=n.std(n.concatenate((a_dops[a_gidx],s_dops[s_gidx])))
        xhat=n.linalg.lstsq(A,m)[0]
        model=n.dot(A,xhat)
        std_est1=n.std(model-m)
        std_est=n.max([std_est0,std_est1])
        post_std=std_est*n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(A),A))))
        print(post_std)
        Z[hi]=xhat[0]
        Z_std[hi]=post_std[0]        
        M[hi]=xhat[1]
        M_std[hi]=post_std[1]


    ho=h5py.File("wind/hor_wind.h5","w")
    ho["zonal"]=Z
    ho["zonal_std"]=Z_std
    ho["meridional"]=M
    ho["meridional_std"]=M_std
    ho["hgt"]=hg
    ho.close()
    plt.plot(Z,hg,label="Zonal",color="C0")
    plt.plot(Z+Z_std,hg,color="C0",alpha=0.5)
    plt.plot(Z-Z_std,hg,color="C0",alpha=0.5)        
    plt.plot(M,hg,label="Meridional",color="C1")
    plt.plot(M+M_std,hg,color="C1",alpha=0.5)
    plt.plot(M-M_std,hg,color="C1",alpha=0.5)
    plt.plot(n.sqrt(Z**2.0+M**2.0),hg,label="Absolute",color="C2")        
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Height (km)")
    plt.ylim([86,102])
    plt.xlim([-100,100])
    plt.legend()    
    plt.tight_layout()
    plt.savefig("figs/hor_wind.png")
    plt.show()

    
def outlier_fit_line(x,y):
    n_m=len(x)
    xp=n.copy(x)
    xp=xp-n.median(x)
    A=n.zeros([n_m,2])
    A[:,0]=1
    A[:,1]=xp
    xhat=n.linalg.lstsq(A,y)[0]
    model=n.dot(A,xhat)
    err_std=n.std(model-y)
    gidx=n.where(n.abs(model-y)<3.0*err_std)[0]
    A2=A[gidx,:]
    y2=y[gidx]
    xp2=xp[gidx]
    xhat=n.linalg.lstsq(A2,y2)[0]
    model2=n.dot(A2,xhat)    
    err_std=n.std(model2-y2)
    gidx2=gidx[n.where(n.abs(model2-y2)<2.0*err_std)[0]]
    return(gidx2)
    
    
def sod_prof_int(L=4e3):
    
    h=h5py.File("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc6/points-1607088638.h5","r")
    p=n.copy(h["ecef_m"][()])
    snr=n.copy(h["SNR_dB"][()]).flatten()
    vel=n.copy(h["dop_vel"][()]).flatten()
    lat=n.copy(h["lat_deg"][()]).flatten()
    h_km=n.copy(h["height_km"][()]).flatten()    
    lon=n.copy(h["lon_deg"][()]).flatten()
    t0=n.copy(h["t0"][()])
    h.close()
    h=h5py.File("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc6/points-1607088648.h5","r")
    p=n.hstack((p,n.copy(h["ecef_m"][()])))
    snr=n.concatenate((snr,n.copy(h["SNR_dB"][()]).flatten()))
    vel=n.concatenate((vel,n.copy(h["dop_vel"][()]).flatten()))
    lat=n.concatenate((lat,n.copy(h["lat_deg"][()]).flatten()))
    lon=n.concatenate((lon,n.copy(h["lon_deg"][()]).flatten()))
    h_km=n.concatenate((h_km,n.copy(h["height_km"][()]).flatten()))

    gidx=outlier_fit_line(lat,h_km)
    lat=lat[gidx]
    lon=lon[gidx]
    vel=vel[gidx]
    h_km=h_km[gidx]
    p=p[:,gidx]
    snr=snr[gidx]    

    plt.subplot(121)
    plt.plot(lat,h_km,".")
    plt.subplot(122)
    plt.plot(lon,h_km,".")
    plt.show()
    
    t0=n.copy(h["t0"][()])
    h.close()
    snr=10**(snr/10.0)
    print(p.shape)
    n_p = p.shape[1]
    p2=n.copy(p)
    vel2=n.copy(vel)
    vel_std=n.copy(vel)    
    for pi in range(n_p):
        idx=[]
        for pi2 in range(n_p):
            dist=n.linalg.norm( p[:,pi]-p[:,pi2])
#            print(dist)
            if dist < L:
                idx.append(pi2)
        idx=n.array(idx,dtype=n.int)
        vel2[pi]=(1.0/(n.sum(snr[idx])))*n.sum(snr[idx]*vel[idx])
        vel_std[pi]=n.std(vel[idx])

    sod_p=jcoord.geodetic2ecef(stc.sod_coords[0], stc.sod_coords[1], stc.sod_coords[2])
    r_ranges=[]
    for pi in range(n_p):
        r_ranges.append(n.linalg.norm(p[:,pi]-sod_p))
    r_ranges=n.array(r_ranges)
                               
    # setup Lambert Conformal basemap.False68.29911309985064, 23.3381106574698
    map_lat=68.25407756526351
    map_lon=24.23106261125355
    
    ho=h5py.File("sod_data/trail_range_dop.h5","w")
    ho["vel"]=vel2
    ho["snr"]=snr
    ho["range"]=r_ranges
    ho["lat"]=lat
    ho["ecef"]=p
    ho["lon"]=lon
    ho["h_km"]=h_km    
    ho.close()

    m = Basemap(width=200e3,
                height=200e3,
                projection='lcc',
                resolution='h',
                lat_0=map_lat,
                lon_0=map_lon)
    # draw coastlines.
    m.drawmapboundary(fill_color="black")
    try:
        m.drawcoastlines(color="white")
    except:
        print("no coastlines")
    m.drawcountries(color="white")
    parallels = n.arange(0.,81,1.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,False],color="white")
    meridians = n.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[False,False,False,True],color="white")
    print("got here")

    
    xlat,ylon=m(lon,lat)
    m.scatter(xlat,ylon,c=vel2,s=1.0,marker="o",vmin=-100,vmax=100,cmap="seismic")
    cb=plt.colorbar()
    cb.set_label("Radial velocity (m/s)")
    plt.savefig("figs/sod_trail_dop_map.png")
    plt.show()


def height_to_latlon():
    """
    k vector in local horizontal coordinate system
    """
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    p=ht["ecef_pos_m"][()]
    lat=ht["lat_wgs84"][()]
    lon=ht["lon_wgs84"][()]        
    hg=ht["h_km_wgs84"][()]

    latp=n.polyfit(hg,lat,3)
    lonp=n.polyfit(hg,lon,3)
    latf=n.poly1d(latp)
    lonf=n.poly1d(lonp)
    return(latf,lonf)


    
def sod_wind_map(vel_col=True):
    ho=h5py.File("sod_data/trail_range_dop.h5","r")
    vel=n.copy(ho["vel"][()])
    snr=n.copy(ho["snr"][()])
    ranges=n.copy(ho["range"][()])
    lat=n.copy(ho["lat"][()])
    p=n.copy(ho["ecef"][()])
    lon=n.copy(ho["lon"][()])
    h_km=n.copy(ho["h_km"][()])
    ho.close()
    
    ho=h5py.File("wind/hor_wind.h5","r")
    w_h0=n.copy(ho["hgt"][()])
    w_m0=n.copy(ho["meridional"][()])
    w_z0=n.copy(ho["zonal"][()])
    w_mf=si.interp1d(w_h0,w_m0)
    w_zf=si.interp1d(w_h0,w_z0)

    dh=3.0
    w_h=n.linspace(86,102,num=10)
    w_m=w_mf(w_h)
    w_z=w_zf(w_h)
#    plt.plot(w_m,w_h)
  #  plt.plot(w_z,w_h)
  #  plt.show()
    ho.close()
    latf,lonf=height_to_latlon()
    # setup Lambert Conformal basemap.False68.29911309985064, 23.3381106574698
    map_lat=68.25407756526351
    map_lon=24.23106261125355
    
    m = Basemap(width=150e3,
                height=150e3,
                projection='lcc',
                resolution='i',
                lat_0=map_lat,
                lon_0=map_lon)
    # draw coastlines.
    m.drawmapboundary(fill_color="black")
    try:
        m.drawcoastlines(color="white")
    except:
        print("no coastlines")
    m.drawcountries(color="white")
    parallels = n.arange(0.,81,0.5)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,False],color="white")
    meridians = n.array([23,24,26])#n.arange(10.,351.,1.)
    m.drawmeridians(meridians,labels=[False,False,False,True],color="white")
    print("got here")

    
    xlat,ylon=m(lon,lat)
    if vel_col:
        m.scatter(xlat,ylon,c=vel,s=1.0,marker="o",vmin=-100,vmax=100,cmap="seismic")
        cb=plt.colorbar()
        cb.set_label("Velocity (m/s)")
        
    else:
        m.scatter(xlat,ylon,c=h_km,s=1.0,marker="o",vmin=60,vmax=105,cmap="jet")
        cb=plt.colorbar()
        cb.set_label("Height (km)")
        

    w_lat=latf(w_h)
    w_lon=lonf(w_h)

    wx,wy=m(w_lon,w_lat)
    ws=4
    u10_rot, v10_rot, wx, wy = m.rotate_vector(ws*w_z,ws*w_m, w_lon+0.05, w_lat+0.05, returnxy=True)
    q1=m.quiver(wx,wy,u10_rot,v10_rot,color="white")
    
#    q1=m.quiver(6818,141315,ws*100.0,0,color="white")
    plt.quiverkey(q1, 0.1,0.05, ws*100.0, '100 m/s',color="white",labelcolor="white")
    plt.tight_layout()

    plt.savefig("figs/sod_trail_wind_map.png")
    plt.show()
    


#def sod_rd_points():
    
    

def sod_prof():
    sod_files=["/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc5/rd-1607088648.h5",
               "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc5/rd-1607088658.h5"]
 #              "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc5/rd-1607088658.h5"]

    #           "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc5/rd-1607088658.h5",
    #          "/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc5/rd-1607088668.h5"]
    
    
    h=h5py.File(sod_files[0],"r")
    print(h.keys())
    rds=n.copy(h["S"][()])
    
    ipp=n.copy(h["ipp_range"][()])
    ranges=n.copy(h["range"][()])+ipp*2
    vels=n.copy(h["vels"][()])
    rds=n.copy(h["S"][()])
    rds[:,:]=0
    h.close()
    n_avg=0.0
    for f in sod_files:
        h=h5py.File(f,"r")
        S0=h["S"][()]
        nf=n.nanmedian(S0)
        S0=(S0-nf)/nf
        S0[S0<=0]=1e-3
        rds+=S0
    n_avg+=1.0

    v0s=[]
    for ri in range(rds.shape[0]):
        rds[ri,:]=rds[ri,:]-n.median(rds[ri,:])
        rds[ri,:]=rds[ri,:]/n.max(rds[ri,:])
        v0try=vels[n.argmax(rds[ri,:])]
        xhat=fit_spec(rds[ri,:],vels,v0try,50.0)
        print(xhat)
        v0s.append(xhat[0])
    v0s=n.array(v0s)
    
    
    rds[rds<=0]=1e-3    
    plt.pcolormesh(vels,ranges,10.0*n.log10(rds/n_avg),vmin=-20,vmax=0)
    plt.plot(v0s,ranges)
    plt.colorbar()
    plt.show()
    
if __name__ == "__main__":
    sod_wind_map(vel_col=False)
    
#loc = EarthLocation(lat=stc.sod_coords[0]*u.deg,lon=stc.sod_coords[1]*u.deg,height=100e3*u.m)
#sod_p=jcoord.geodetic2ecef(stc.sod_coords[0], stc.sod_coords[1], 100e3)
#loc=n.array([loc.x.value,loc.y.value,loc.z.value])
#north = AltAz(az=0*u.deg, alt=0*u.deg,
#bae              location = loc)
#sk=SkyCoord(north,obstime="2020-12-04")
#print(sk.itrs)
#east = AltAz(az=90*u.deg, alt=0*u.deg,
#              location = loc)
#ske=SkyCoord(east,obstime="2020-12-04")
#print(ske.itrs)
#no=jcoord.enu2ecef(stc.sod_coords[0], stc.sod_coords[1], 100e3, 0, 1, 0)
#ea=jcoord.enu2ecef(stc.sod_coords[0], stc.sod_coords[1], 100e3, 1, 0, 0)
#print(jcoord.ecef2geodetic(sod_p[0], sod_p[1], sod_p[2]))


invert_wind()

#and_range_to_height()          
#plot_and_prof()
#sod_prof()
#sod_prof_int()
