import numpy as n
import matplotlib.pyplot as plt
import h5py
#import pyglow
from datetime import datetime
import scipy.interpolate as si
import scipy.optimize as sio
import scipy.constants as c

import msise00


def rho_a(h):
    # Atmospheric density, as a function of height, using the MSIS model
    # use kg/m^3
    lat=69.0
    lon=19.0
    nh=len(h)
    rho=n.zeros(nh)

    mmass=n.zeros(nh)
    
    dn=datetime(2017,12,4,13,30,0)

    for i in range(nh):
        N=msise00.run(time=dn,altkm=h[i],glat=lat,glon=lon)
#        print(N.keys())
        rHe=N["He"][0,0,0,0]
        rO=N["O"][0,0,0,0]
        rO2=N["O2"][0,0,0,0]        
        rN2=N["N2"][0,0,0,0]
        rAr=N["Ar"][0,0,0,0]
        rH=N["H"][0,0,0,0]
        rN=N["N"][0,0,0,0]
        rTot=N["Total"][0,0,0,0]
        N=N["Total"]

        mean_amu = (4.002*rHe + 15.9994*rO + 2*15.9994*rO2 + rN2*2*14.0067 + 39.948*rAr + rH*1.007 + 14.0067*rN)/(rHe + rO + rO2 + rN2 + rAr + rH + rN)
        print( "mean molecular mass %f (amu)"%(mean_amu))
        mmass[i]=mean_amu*c.u
        print("%f %g"%(h[i],N[0,0,0,0]))
        rho[i]=N[0,0,0,0]
    return(rho)
    

def dyn_mass(md,rhoa,v,A=1.209,gamma=1.0,rho_m=3000.0):
    a=(gamma*rhoa*v**2.0/md)*A*(md/rho_m)**(2.0/3.0)
    return(a)

def dyn_mass0(rhoa,v,a,A=1.209,gamma=1.0,rho_m=3000.0):
    """ 
     Halliday 1996 / Gritsevich 2008
     initial guess
    """
    # 2L x 3L x 5L brick
    m_d = -(3.75/rho_m**2.0)*( (rhoa*v**2.0)/ (2.0*a))**3.0
    return(m_d)

# https://www.meteornews.net/2018/04/02/detailed-analysis-of-the-fireball-20160317_031654-over-united-kingdom/
def fit_exp(rho,hg):
    def model(x):
        a0=10**x[0]
        a1=10**x[1]        
        h0=10**x[2]
        return(a0*n.exp(-a1*(hg-h0)))
    def ss(x):
        m=model(x)
        s=n.sum(n.abs(rho-m)**2.0)
        print(s)
        return(s)
    #    a0*n.exp(-a1*(60-60)) = 1e-4
    #    1e-4*n.exp(-a1*(
    xhat=sio.fmin(ss,[-4,1.0,n.log10(60)])
    #    plt.plot(model(xhat),hg)
    #   plt.plot(rho,hg,".")
    #  plt.show()
    return(model(xhat))

    
# Surface temp, meteor density, Surface temperature of 2500 K


h=h5py.File("state_vector/chain.h5","r")
v=n.copy(h["vels"][()])
t=n.copy(h["t"][()])
hg=n.copy(h["h_km"][()])

hg_mean_km=n.copy(h["h_mean_km"][()])
hg_std_km=n.copy(h["h_std_km"][()])

dp=n.copy(h["dp"][()])
h.close()

#print(v.shape)
#print(len(hg))
#fig=plt.figure()
#ax=plt.axes()
#ax.set_xscale("log")
#ax.set_yscale("log")
#plt.plot(n.mean(v,axis=0)/1e3,n.log10(hg_mean_km/1e3),"o")
mean_vel = n.mean(v,axis=0)/1e3
std_vel = n.std(v,axis=0)/1e3
mean_height= hg_mean_km/1e3

ho=h5py.File("pajala.h5","w")
ho["mean_vel"]=mean_vel
ho["std_vel"]=std_vel
ho["mean_height_km"]=mean_height
ho["std_height_km"]=hg_std_km/1e3
ho.close()
fo = open("alphabeta.csv","w")
fo.write("height,D_DT_geo,D_DT_fitted\n")
for i in range(len(mean_vel)):
    fo.write("%1.5f,%1.5f,%1.5f\n"%(mean_height[i]*1e3,mean_vel[i]*1e3,mean_vel[i]*1e3))
fo.close()
plt.errorbar(n.mean(v,axis=0)/1e3,hg_mean_km/1e3,xerr=2*n.std(v,axis=0)/1e3,yerr=2*hg_std_km/1e3,fmt=".")
plt.xlabel("Velocity (km/s)")
plt.ylabel("Height (km)")
plt.show()

md4s=[]
md2s=[]
md1s=[]
plot=True
rhoa=rho_a(hg)
rhoaf=fit_exp(rhoa,hg)
#plt.plot(rhoaf,hg)
#plt.plot(rhoa,hg,".")
#plt.show()

#print(v.shape)
#for i in range(v.shape[0]):


M=n.zeros([3,dp.shape[0],len(hg)])

for i in range(dp.shape[0]):
    # acceleration 
    a = -dp[i,0]*n.exp(dp[i,1]*t)
    md_4=dyn_mass0(rhoa,v[i,:],a,rho_m=4000.0)    
    md_2=dyn_mass0(rhoa,v[i,:],a,rho_m=1700.0)
    md_1=dyn_mass0(rhoa,v[i,:],a,rho_m=600.0)

    md4s.append(n.max(md_4))
    md2s.append(n.max(md_2))
    md1s.append(n.max(md_1))

    M[0,i,:]=md_4
    M[1,i,:]=md_2
    M[2,i,:]=md_1

plt.fill_betweenx(hg,n.median(M[0,:,:],axis=0)-n.std(M[0,:,:],axis=0),n.median(M[0,:,:],axis=0)+n.std(M[0,:,:],axis=0),color="C0",alpha=0.5)
plt.plot(n.median(M[0,:,:],axis=0),hg,color="C0",label="4 g/cm$^{3}$")

plt.fill_betweenx(hg,n.median(M[1,:,:],axis=0)-n.std(M[1,:,:],axis=0),n.median(M[1,:,:],axis=0)+n.std(M[1,:,:],axis=0),color="C1",alpha=0.5)
plt.plot(n.median(M[1,:,:],axis=0),hg,color="C1",label="1.7 g/cm$^{3}$")

plt.fill_betweenx(hg,n.median(M[2,:,:],axis=0)-n.std(M[2,:,:],axis=0),n.median(M[2,:,:],axis=0)+n.std(M[2,:,:],axis=0),color="C2",alpha=0.5)
plt.plot(n.median(M[2,:,:],axis=0),hg,color="C2",label="0.6 g/cm$^{3}$")
plt.xscale("log")
#plt.semilogx(n.median(M[0,:,:],axis=0),hg,color="C0")
#plt.semilogx(n.median(M[0,:,:],axis=0)+2.0*n.std(M[0,:,:],axis=0),hg,color="C0",alpha=)
#plt.semilogx(n.median(M[0,:,:],axis=0)-2.0*n.std(M[0,:,:],axis=0),hg,color="C0")

#plt.xlim([1e-3,1])
#plt.show()
#    if plot:
 #       if i==0:
  #          plt.semilogx(md_1*1e3,hg,alpha=0.2,color="C0",label="0.6 g/cm$^3$")
   #         plt.semilogx(md_2*1e3,hg,alpha=0.2,color="C1",label="1.7 g/cm$^3$")
    #        plt.semilogx(md_4*1e3,hg,alpha=0.2,color="C2",label="4.0 g/cm$^3$")
        

print("4000 kg/m^3 %1.3f +/- %1.4f [%1.2g,%1.2g]"%(n.mean(md4s),n.std(md4s),n.min(md4s),n.max(md4s)))
print("1700 kg/m^3 %1.3f +/- %1.4f [%1.2g,%1.2g]"%(n.mean(md2s),n.std(md2s),n.min(md2s),n.max(md2s)))
print("600 kg/m^3 %1.3f +/- %1.4f [%1.2g,%1.2g]"%(n.mean(md1s),n.std(md1s),n.min(md1s),n.max(md1s)))
#    plt.plot(4.0*dyn_mass0(rho_a(hg),v[i,:],a,rho_m=800.0),hg,alpha=0.2,color="gray")
 #   plt.plot(1.25*dyn_mass0(rho_a(hg),v[i,:],a,rho_m=4000.0),hg,alpha=0.2,color="red")
plt.legend()
if plot:
    plt.title("Dynamic mass estimate")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Altitude (km)")
    plt.show()

