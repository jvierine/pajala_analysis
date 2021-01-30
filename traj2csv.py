import h5py
import stuffr
import jcoord


h=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
print(h.keys())

labels=["Time (UTC)",
        "Time (unix)",
        "ITRS x (m)",
        "ITRS y (m)",
        "ITRS z (m)",
        "WGS84 lat (deg)",
        "WGS84 lon (deg)",
        "WGS84 height (km)"]
print(len(labels))
print("%0.18s, %.18s, %.18s, %.18s, %.18s, %.18s, %.18s, %.18s"%(labels[0],labels[1],labels[2],labels[3],labels[4],labels[5],labels[6],labels[7]))

n_t=len(h["t_unix"][()])
tu=h["t_unix"][()]
ep=h["ecef_pos_m"][()]
lat=h["lat_wgs84"][()]
lon=h["lon_wgs84"][()]
h=h["h_km_wgs84"][()]
for i in range(n_t):
    print("%s, %1.2f, %1.2f, %1.2f, %1.2f, %1.3f, %1.3f, %1.3f"%(stuffr.unix2datestr(tu[i]),tu[i],ep[i,0],ep[i,1],ep[i,2],lat[i],lon[i],h[i]))

