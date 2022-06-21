import h5py
import numpy as n
import matplotlib.pyplot as plt

import pyglow
from datetime import datetime
import scipy.interpolate as si

def rho_a(h):
    lat=69.0
    lon=19.0
    nh=len(h)
    rho=n.zeros(nh)
    dn=datetime(2017,12,4,13,30,0)

    for i in range(nh):
        pt=pyglow.Point(dn,lat,lon,h[i])
        pt.run_msis()
        rho[i]=pt.rho*1e6/1e3

    return(rho)

h=h5py.File("wind/hor_wind.h5","r")
print(h.keys())
hgt=n.copy(h["hgt"][()])
zonal=n.copy(h["zonal"][()])
zonal_std=n.copy(h["zonal_std"][()])
meridional=n.copy(h["meridional"][()])
meridional_std=n.copy(h["meridional_std"][()])
gidx=n.where( (hgt > 86) & (hgt < 102) )[0]
hgt=hgt[gidx]
zonal=zonal[gidx]
zonal_std=zonal_std[gidx]
meridional=meridional[gidx]
meridional_std=meridional_std[gidx]

#gidx=n.where( (hgt < 93.5) | (hgt > 94.5) )[0]
#meridional=meridional[gidx]
#meridional_std=meridional_std[gidx]

zp=n.polyfit(hgt*1e3,zonal,50)
zf=n.poly1d(zp)
mp=n.polyfit(hgt*1e3,meridional,50)
mf=n.poly1d(mp)

du_dz = n.zeros(len(hgt))
dz=100.0
du_dz = n.sqrt( ((zf(hgt*1e3+0.5*dz)-zf(hgt*1e3-0.5*dz))/dz)**2.0 + ( (mf(hgt*1e3+0.5*dz)-mf(hgt*1e3-0.5*dz))/dz)**2.0)

ihgt=n.copy(hgt)
ihgt[0]=ihgt[0]-10.0
ihgt[-1]=ihgt[-1]+10.0
zf=si.interp1d(ihgt*1e3,zonal)
mf=si.interp1d(ihgt*1e3,meridional)
du_dz = n.sqrt( ((zf(hgt*1e3+0.5*dz)-zf(hgt*1e3-0.5*dz))/dz)**2.0 + ( (mf(hgt*1e3+0.5*dz)-mf(hgt*1e3-0.5*dz))/dz)**2.0)

N_max=(2.0*n.pi/(4*60.0))
if True:
    plt.errorbar(zonal,hgt,xerr=zonal_std,fmt="o",color="C0",label="Zonal ($u$)")
    plt.plot(zf(hgt*1e3),hgt,color="C0")

    
    plt.errorbar(meridional,hgt,xerr=meridional_std,fmt="o",color="C1",label="Meridional ($v$)")
    plt.plot(mf(hgt*1e3),hgt,color="C1")
    plt.plot(du_dz*1e3,hgt,color="C2",label="$\sqrt{(\partial_z u_h)^2}$")
    plt.plot(10.0*N_max**2.0/(du_dz)**2.0,hgt,color="C3",label="$10 \\times \mathrm{Ri}$")
    plt.legend()
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Height (km)")

    plt.grid()
    plt.tight_layout()
    plt.xlim([-80,80])    
    plt.savefig("figs/hor_shear.png")

    plt.show()

dz=6e3
rhos=rho_a(n.array([90+dz/1e3,90]))
rho9=rho_a(n.array([92]))

#dr_dz = (rhos[1]-rhos[0])/dz
#du_dz2 = ((40.0+45.0)/6e3)**2.0 + ((9.0+13.0)/6e3)**2.0
#print(dr_dz)
#print(du_dz2)
#print(rho9)
#print(rhos)
#print((9.81/rho9[0])*(dr_dz/du_dz2))



plt.plot(N_max**2.0/(du_dz/1e3)**2.0,hgt,".")
plt.ylabel("Height (km)")
plt.xlabel("Richardson number (10x)")
plt.show()
