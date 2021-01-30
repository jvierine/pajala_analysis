import mean_free_path as mfp
import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as si
import scipy.constants as c

def ne_model(rcs=17e3,h=92.0,fr=36.9e6):
    lam=mfp.mfp(h)
    r0_min=1.0*lam
    r0_max=1.0*lam
    rp = n.sqrt(rcs/n.pi)
    n0_min = ((fr/9.0)**2.0)*(1+(rp/r0_min)**2.0)
    n0_max = ((fr/9.0)**2.0)*(1+(rp/r0_max)**2.0)
    
    return(n0_min,n0_max,r0_min,r0_max,rp)

def line_density(rcs=17e3,h=92.0,fr=36.9e6,R_max=500.0):
    
    n00,n01,r00,r01,rp=ne_model(rcs=rcs,h=h,fr=fr)
    r=n.linspace(0,R_max,num=1000000)
    dr=n.diff(r)[0]
    ne0=n00/(1.0+(r/r00)**2.0)
    ne1=n01/(1.0+(r/r01)**2.0)
    q_max=n.sum(2.0*n.pi*r*ne0)*dr
    q_min=n.sum(2.0*n.pi*r*ne1)*dr
    return(q_max,q_min,r00,r01,rp)

def mass_est(mu=24.0*c.atomic_mass,beta=0.2,v=28e3):
    ho=h5py.File("rcs/rcs_est.h5","r")
    s_rcs=ho["s_rcs"][()]
    s_rcs_h=ho["s_rcs_h"][()]    
    a_rcs=ho["a_rcs"][()]
    a_rcs_h=ho["a_rcs_h"][()]
    plt.plot(s_rcs,s_rcs_h,".")
    plt.plot(a_rcs,a_rcs_h,".")

    rcs_fun = si.interp1d(n.concatenate((s_rcs_h,a_rcs_h)),n.concatenate((s_rcs,a_rcs)))
    h=n.linspace(106,85,num=200)
    
    t = n.linspace(0,6.0,num=200)
    
    hfun=si.interp1d(t,h)
    plt.plot(rcs_fun(h),h)
    plt.show()

    plt.plot(t,hfun(t))
    plt.show()

    dt=n.diff(t)[0]
    M=0.0
    for ti in range(len(t)):
        print(rcs_fun(hfun(t[ti])))
        q0,q1,r0,r1,rp=line_density(rcs=rcs_fun(hfun(t[ti])),
                       h=hfun(t[ti]),
                       fr=36.9e6)
        print("q %1.2f r %1.2f rp %1.2f"%(n.log10(q0),r0,rp))
        M+=dt*q0*mu*v/beta
    print("done")
    print(M)
    
    
if __name__ == "__main__":
    mass_est()


#    q_max=n.sum(2.0*n.pi*r*ne0)*dr
 #   q_min=n.sum(2.0*n.pi*r*ne1)*dr
 #   print(q_max)
#    print(q_min)
    
#    plt.plot(r,ne0,label="min")
 #   plt.plot(r,ne1,label="max")
  #  plt.legend()
   # plt.axhline(36.9e6)
    #plt.show()
    
