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


import trail_profile as tp

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
#    def height_to_latlon():
    latf,lonf=tp.height_to_latlon()
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
    parallels = n.arange(0.,81,1.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,False],color="white")
    meridians = n.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[False,False,False,True],color="white")
    print("got here")

    
    xlat,ylon=m(lon,lat)
    if vel_col:
        m.scatter(xlat,ylon,c=vel,s=1.0,marker="o",vmin=-100,vmax=100,cmap="seismic")
        cb=plt.colorbar()
        cb.set_label("Velocity (m/s)")
        
    else:
        m.scatter(xlat,ylon,c=h_km,s=1.0,marker="o",vmin=60,vmax=105,cmap="plasma")
        cb=plt.colorbar()
        cb.set_label("Height (km)")
        

    w_lat=latf(w_h)
    w_lon=lonf(w_h)

    if False:
        w_dl=0.1
        for wi in range(len(w_h)):
            wx0,wy0=m(w_lon[wi],w_lat[wi])
            wx1,wy1=m(w_lon[wi]+w_dl*w_z[wi]/100.0,w_lat[wi]+w_dl*w_m[wi]/100.0)
            m.plot([wx0,wx1],[wy0,wy1],color="white")
        
    if True:
        wx,wy=m(w_lon,w_lat)
        ws=4
        u10_rot, v10_rot, wx2, wy2 = m.rotate_vector(ws*w_z,ws*w_m, w_lon, w_lat, returnxy=True)
        q1=m.quiver(wx,wy,u10_rot,v10_rot,color="white")
        
        #    q1=m.quiver(6818,141315,ws*100.0,0,color="white")
        plt.quiverkey(q1, 0.1,0.05, ws*100.0, '100 m/s',color="white",labelcolor="white")

        
    plt.savefig("figs/sod_trail_wind_map.png")
    plt.show()


sod_wind_map()
