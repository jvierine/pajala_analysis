import h5py
import numpy as n
import matplotlib.pyplot as plt
import jcoord
import radiant_est

from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.time import Time

import scam

def itrs_model(pts,ra,dec,p0,t0,plot=True):
    """
    parameters base position p0 (lat,lon,h)
    right ascension and declination of radiant
    time of measurement
    """
    # use center point
    obs_time=Time(t0,format="unix")
    n_m=pts.shape[1]
    cp=n.mean(pts,axis=1)
    loc = EarthLocation(x=cp[0]*u.m,y=cp[1]*u.m,z=cp[2]*u.m)
    #loc0=n.array(loc.itrs.cartesian.xyz)
    cp.shape=(3,1)
    pts = pts - cp
    model_pts=n.zeros([3,n_m])
    def model(x):
        ra0=x[0]
        dec0=x[1]
        
        sky = SkyCoord(ra = ra0*u.deg,
                       dec = dec0*u.deg,
                       obstime = obs_time,
                       frame = 'icrs',
                       location = loc)     
        u0=n.array(sky.itrs.cartesian.xyz)
        
        t = pts[0,:]*u0[0] + pts[1,:]*u0[1] + pts[2,:]*u0[2]
        model_pts[0,:]=t*u0[0]
        model_pts[1,:]=t*u0[1]
        model_pts[2,:]=t*u0[2]
        
        return(model_pts)
    
    pm=model([ra,dec])
    
    if plot:
        plt.plot(pm[0,:],pm[1,:])
        plt.plot(pts[0,:],pts[1,:],".")
        plt.plot([0],[0],"x",color="red")
        plt.show()
        
    res=pm-pts
    xv=n.var(res[0,:])
    yv=n.var(res[1,:])
    zv=n.var(res[2,:])    

    def ss(x):
        m=model(x)
        s=(1.0/xv)*n.sum(n.abs(m[0,:]-pts[0,:])**2.0)+(1.0/yv)*n.sum(n.abs(m[1,:]-pts[1,:])**2.0)+(1.0/zv)*n.sum(n.abs(m[2,:]-pts[2,:])**2.0)
        return(s)

    chain=scam.scam(ss,n.array([ra,dec]),step=[0.1,0.1],n_iter=10000,debug=False)
    mra=n.mean(chain[:,0])
    mdec=n.mean(chain[:,1])
    ra_std=n.std(chain[:,0])
    dec_std=n.std(chain[:,1])
    print("RA %1.2f +/- %1.2f (2-sigma)"%(mra,2*ra_std))
    print("Dec %1.2f +/- %1.2f (2-sigma)"%(mdec,2*dec_std))
    if plot:
        plt.hist2d(chain[:,0],chain[:,1],bins=10)
        plt.title("Radian location")
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")    
        plt.show()

        pm=model([mra,mdec])
        plt.plot(pm[0,:],pm[1,:])
        plt.plot(pts[0,:],pts[1,:],".")
        plt.plot([0],[0],"x",color="red")
        plt.show()
    return(mra,ra_std,mdec,dec_std)

def fit_line(lat,
             lon,
             h,
             min_height=70.0,
             max_height=120,
             plot=True):
    """
    Estimate start and end point of line
    This is a simple/stupid radiant estimator,
    which is somewhat robust and has outlier removal. 
    This should be fed into into itrs_model to get a
    error bars.
    """
    pts=jcoord.geodetic2ecef(lat, lon, h*1e3)

    pf0=n.polyfit(h,lat,1)
    latf=n.poly1d(pf0)
    
    stdest=n.std(latf(h)-lat)

    res=latf(h)-lat
    gidx=n.where( (n.abs(res)<3.0*stdest) & (h > min_height) & (h < max_height) )[0]
    
    pf0=n.polyfit(h[gidx],lat[gidx],1)
    latf=n.poly1d(pf0)
    pf0=n.polyfit(h[gidx],lon[gidx],1)
    lonf=n.poly1d(pf0)
    
    if plot:
        plt.subplot(121)
        plt.plot(lat[gidx],h[gidx],".")
        plt.xlabel("Lat (deg)")
        plt.ylabel("Height (km)")        
        plt.plot(latf(h[gidx]),h[gidx])    
        plt.subplot(122)
        plt.plot(lon[gidx],h[gidx],".")
        plt.plot(lonf(h[gidx]),h[gidx])
        plt.xlabel("Lon (deg)")
        plt.ylabel("Height (km)")        
        plt.show()
        
    p0=jcoord.geodetic2ecef(latf(h[0]), lonf(h[0]), h[0]*1e3)
    p1=jcoord.geodetic2ecef(latf(h[-1]), lonf(h[-1]), h[-1]*1e3)

    u0 = (p1-p0)/n.linalg.norm(p1-p0)

    #
    # model p(t) = p0 + t*u0 + xi
    #
    # px(t) = p0x + t*u0x + xix
    # py(t) = p0y + t*u0y + xiy
    # pz(t) = p0z + t*u0z + xiz
    #
    return(p0,p1,gidx)


if __name__ == "__main__":    
        h=h5py.File("data/2020-12-04-meter_radar.h5","r")

        p0,p1,gidx=fit_line(n.copy(h[("lat_deg")]),n.copy(h[("lon_deg")]),n.copy(h[("height_km")]))
        uvec=p0-p1
        u0=uvec/n.linalg.norm(uvec)
        radiant=radiant_est.get_radiant(p0,h[("t0")],u0)
        ra=radiant.icrs.ra.deg
        dec=radiant.icrs.dec.deg
        print("Initial guess RA/Dec %1.2f/%1.2f"%(ra,dec))

        
        t0=h[("t0")]
        lat=h[("lat_deg")][gidx]
        lon=h[("lon_deg")][gidx]
        hkm=h[("height_km")][gidx]
        pts=n.copy(h[("ecef_m")])
        pts=pts[:,gidx]


        ra,dec,ra_std,dec_std=itrs_model(pts,ra,dec,[lat[0],lon[0],hkm[0]*1e3],t0)


