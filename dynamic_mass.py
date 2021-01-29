import numpy as n
import matplotlib.pyplot as plt
import h5py
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
    

def dyn_mass(md,rhoa,v,A=1.209,gamma=1.0,rho_m=3000.0):
    a=(gamma*rhoa*v**2.0/md)*A*(md/rho_m)**(2.0/3.0)
    return(a)

def dyn_mass0(rhoa,v,a,A=1.209,gamma=1.0,rho_m=3000.0):
    """ 
     Halliday 1996 / Gritsevich 2008
    initial guess
    """
#    m_d = -((gamma*A)**3.0/rho_m**2.0)*(rhoa*v**2.0)**3.0/a**3.0
    #A=15*L**2.0
    #m=30.0*rho_m*L**3.0
    m_d = -(3.75/rho_m)*(rhoa*v**2.0/(2.0*a))**3.0
    return(m_d)

# https://www.meteornews.net/2018/04/02/detailed-analysis-of-the-fireball-20160317_031654-over-united-kingdom/



h=h5py.File("state_vector/chain.h5","r")
v=n.copy(h["vels"][()])
t=n.copy(h["t"][()])
hg=n.copy(h["h_km"][()])
dp=n.copy(h["dp"][()])
h.close()

md4s=[]
md2s=[]
md1s=[]
plot=False
for i in range(200):
    a = -dp[i,0]*n.exp(dp[i,1]*t)
    md_4=dyn_mass0(rho_a(hg),v[i,:],a,rho_m=4000.0)    
    md_2=dyn_mass0(rho_a(hg),v[i,:],a,rho_m=2000.0)
    md_1=dyn_mass0(rho_a(hg),v[i,:],a,rho_m=1000.0)
    md4s.append(md_4)
    md2s.append(md_2)
    md1s.append(md_1)
    if plot:
        plt.plot(md_1,hg,alpha=0.2,color="C0",label="1 g/cm$^3$")
        plt.plot(md_2,hg,alpha=0.2,color="C1",label="2 g/cm$^3$")
        plt.plot(md_4,hg,alpha=0.2,color="C2",label="4 g/cm$^3$")    

print("4000 kg/m^3 %1.2f +/- %1.2f"%(n.mean(md4s),n.std(md4s)))
print("2000 kg/m^3 %1.2f +/- %1.2f"%(n.mean(md2s),n.std(md2s)))
print("1000 kg/m^3 %1.2f +/- %1.2f"%(n.mean(md1s),n.std(md1s)))
#    plt.plot(4.0*dyn_mass0(rho_a(hg),v[i,:],a,rho_m=800.0),hg,alpha=0.2,color="gray")
 #   plt.plot(1.25*dyn_mass0(rho_a(hg),v[i,:],a,rho_m=4000.0),hg,alpha=0.2,color="red")
if plot:
    plt.show()

