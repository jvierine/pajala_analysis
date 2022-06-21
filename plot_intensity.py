import h5py
import matplotlib.pyplot as plt
import numpy as n
import scipy.interpolate as sint

h=h5py.File("camera_data/ski_int.h5","r")
t=n.copy(h["t"][()])
I=n.copy(h["I"][()])
h.close()

h=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
tt=n.copy(h["t_unix"][()])
h_km=n.copy(h["h_km_wgs84"][()])
h.close()

tt[0]=tt[0]-3600
tt[-1]=tt[-1]+3600
hfun=sint.interp1d(tt,h_km)

print(len(t))
print(len(I))
I=I-I[0]

radar_idx=n.where(hfun(t) > 90)[0]
print(n.sum(I[radar_idx])/n.sum(I))

plt.plot(I,hfun(t),label="Video intensity")
plt.xlabel("Pixel intensity (power in arb. units)")
plt.ylabel("Height (km)")
plt.legend()
plt.show()

