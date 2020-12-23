import jcoord
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.time import Time
import numpy as n

def get_radiant(p0,t0,u0):
    """
    initial position (ecef, itrs)
    time (unix)
    u0 unit vector indication velocity vector of meteor
    """
    # tbd: convert to astropy
    llh=jcoord.ecef2geodetic(p0[0], p0[1], p0[2])
    llh2=jcoord.ecef2geodetic(p0[0]-1e3*u0[0], p0[1]-1e3*u0[1], p0[2]-1e3*u0[2])
    aar=jcoord.geodetic_to_az_el_r(llh[0],llh[1],llh[2], llh2[0], llh2[1], llh2[2])

    loc = EarthLocation(lat=llh[0]*u.deg, lon=llh[1]*u.deg, height=llh[2]*u.m)
    obs_time=Time(t0,format="unix")
    aa = AltAz(location=loc, obstime=obs_time)
    sky = SkyCoord(alt = aar[1]*u.deg, az = aar[0]*u.deg, obstime = obs_time, frame = 'altaz', location = loc)
    return(sky)


if __name__ == "__main__":
    p0=n.array([2136665.79696413,1006114.85447708,6013794.45627783])
    t0=1607088637.4973311
    u0=n.array([ 0.8145839,  -0.29389749, -0.50007734])
    r=get_radiant(p0,t0,u0)
    print(r.icrs)
