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
    
def obs_map(L=4e3,
            map_lat=68.25407756526351,
            map_lon=21.9):
    """
    """
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    p=ht["ecef_pos_m"][()]
    lat=ht["lat_wgs84"][()]
    lon=ht["lon_wgs84"][()]        
    hg=ht["h_km_wgs84"][()]

    ski_p=jcoord.geodetic2ecef(stc.ski_coords[0], stc.ski_coords[1], stc.ski_coords[2])
    sod_p=jcoord.geodetic2ecef(stc.sod_coords[0], stc.sod_coords[1], stc.sod_coords[2])
    and_p=jcoord.geodetic2ecef(stc.and_coords[0], stc.and_coords[1], stc.and_coords[2])
    sor_p=jcoord.geodetic2ecef(stc.sor_coords[0], stc.sor_coords[1], stc.sor_coords[2])

    mid_p = 0.5*(ski_p + sod_p)
    mid_llh=jcoord.ecef2geodetic(mid_p[0],mid_p[1],mid_p[2])
    
    m = Basemap(width=500e3,
                height=500e3,
                projection='lcc',
                resolution='i',
                lat_0=map_lat,
                lon_0=map_lon)
    # draw coastlines.
    m.drawmapboundary(fill_color="white")
    #    try:
    m.drawcoastlines(color="grey")
    #except:
    #       print("no coastlines")
    m.drawcountries(color="black")
    parallels = n.arange(0.,81,1.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,False],color="grey")
    meridians = n.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[False,False,False,True],color="grey")
    print("got here")


    # cam view
    xlat,ylon=m([lon[0],stc.ski_coords[1]],[lat[0],stc.ski_coords[0]])
    m.plot(xlat,ylon,color="gray",zorder=1)
    xlat,ylon=m([lon[-1],stc.ski_coords[1]],[lat[-1],stc.ski_coords[0]])
    m.plot(xlat,ylon,color="gray",zorder=1)
    
    # cam view
    xlat,ylon=m([lon[0],stc.sor_coords[1]],[lat[0],stc.sor_coords[0]])
    m.plot(xlat,ylon,color="gray",zorder=1)
    xlat,ylon=m([lon[-1],stc.sor_coords[1]],[lat[-1],stc.sor_coords[0]])
    m.plot(xlat,ylon,color="gray",zorder=1)

    # iono-path view
    xlat,ylon=m([stc.ski_coords[1],stc.sod_coords[1]],[stc.ski_coords[0],stc.sod_coords[0]])
    m.plot(xlat,ylon,color="red",zorder=1)

    
    
    met_lon = [stc.sod_coords[1],stc.and_coords[1]]
    met_lat = [stc.sod_coords[0],stc.and_coords[0]]    
    xlat,ylon=m(met_lon,met_lat)
    m.scatter(xlat,ylon,marker="o",label="Meteor radars",color="black",s=120,zorder=2)

    cam_lon = [stc.ski_coords[1],stc.sor_coords[1]]
    cam_lat = [stc.ski_coords[0],stc.sor_coords[0]]
    xlat,ylon=m(cam_lon,cam_lat)
    m.scatter(xlat,ylon,marker="o",label="Camera",color="blue",s=120,zorder=2)

    ion_lon = [stc.sod_coords[1],stc.ski_coords[1]]
    ion_lat = [stc.sod_coords[0],stc.ski_coords[0]]
    xlat,ylon=m(ion_lon,ion_lat)
    m.scatter(xlat,ylon,marker="D",label="Ionosonde TX/RX",color="white",edgecolor="black",linewidth=1,zorder=2)
    
    ion_lon = [stc.ski_coords[1]]
    ion_lat = [stc.ski_coords[0]]
    xlat,ylon=m(ion_lon,ion_lat)
    m.scatter(xlat,ylon,marker="D",label="Ionosonde RX",color="red",edgecolor="black",linewidth=1,zorder=2)

    

    inf_lon = [stc.bar_coords[1],20.2253,stc.sod_coords[1]]
    inf_lat = [stc.bar_coords[0],67.8558,stc.sod_coords[0]]
    xlat,ylon=m(inf_lon,inf_lat)
    m.scatter(xlat,ylon,marker="D",label="Infrasound",color="orange",edgecolor="black",linewidth=1,zorder=2)

    # sodankylä bearing
    llh_is=jcoord.az_el_r2geodetic(stc.sod_coords[0],stc.sod_coords[1], 0.0, 315.0, 0.0, 600e3)
    xlat,ylon=m([stc.sod_coords[1],llh_is[1]],[stc.sod_coords[0],llh_is[0]])
    m.plot(xlat,ylon,color="orange",zorder=1)

    
    # inf-path view
    llh_is=jcoord.az_el_r2geodetic(stc.bar_coords[0],stc.bar_coords[1], 0.0, 115.0, 0.0, 600e3)
    xlat,ylon=m([stc.bar_coords[1],llh_is[1]],[stc.bar_coords[0],llh_is[0]])
    m.plot(xlat,ylon,color="orange",zorder=1)

    if False:
        # inf-path view
        llh_is=jcoord.az_el_r2geodetic(stc.bar_coords[0],stc.bar_coords[1], 0.0, 115.0-5, 0.0, 600e3)
        xlat,ylon=m([stc.bar_coords[1],llh_is[1]],[stc.bar_coords[0],llh_is[0]])
        m.plot(xlat,ylon,color="orange",zorder=1)
        # inf-path view
        llh_is=jcoord.az_el_r2geodetic(stc.bar_coords[0],stc.bar_coords[1], 0.0, 115.0+5, 0.0, 600e3)
        xlat,ylon=m([stc.bar_coords[1],llh_is[1]],[stc.bar_coords[0],llh_is[0]])
        m.plot(xlat,ylon,color="orange",zorder=1)
        
        # inf-path view
        llh_is=jcoord.az_el_r2geodetic(stc.bar_coords[0],stc.bar_coords[1], 0.0, 115.0-10, 0.0, 600e3)
        xlat,ylon=m([stc.bar_coords[1],llh_is[1]],[stc.bar_coords[0],llh_is[0]])
        m.plot(xlat,ylon,color="orange",zorder=1)
        # inf-path view
        llh_is=jcoord.az_el_r2geodetic(stc.bar_coords[0],stc.bar_coords[1], 0.0, 115.0+10, 0.0, 600e3)
        xlat,ylon=m([stc.bar_coords[1],llh_is[1]],[stc.bar_coords[0],llh_is[0]])
        m.plot(xlat,ylon,color="orange",zorder=1)
        
    
    sod_lon = stc.sod_coords[1]
    sod_lat = stc.sod_coords[0]
    xlat,ylon=m(sod_lon-1.0,sod_lat-0.4)
    plt.text(xlat,ylon,"Sodankylä")

    sor_lon = stc.sor_coords[1]
    sor_lat = stc.sor_coords[0]
    xlat,ylon=m(sor_lon-0.5,sor_lat-0.3)
    plt.text(xlat,ylon,"Sørreisa")

    ski_lon = stc.ski_coords[1]
    ski_lat = stc.ski_coords[0]
    xlat,ylon=m(ski_lon+0.2,ski_lat+0.16)
    plt.text(xlat,ylon,"Skibotn")

    and_lon = stc.and_coords[1]
    and_lat = stc.and_coords[0]
    xlat,ylon=m(and_lon-0.2,and_lat+0.18)
    plt.text(xlat,ylon,"Andenes")
    


    xlat,ylon=m(lon,lat)
    m.scatter(xlat,ylon,marker="o",c=hg,vmin=60,vmax=106,zorder=2)
    cb=plt.colorbar()
    cb.set_label("Height (km)")


    
    plt.legend(loc=3)
    plt.tight_layout()
 #   cb=plt.colorbar()
#    cb.set_label("Radial velocity (m/s)")
    plt.savefig("figs/obs_map.png")
    plt.show()


obs_map()
