import numpy as n
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as so
import radiant_est
import scam

import OD_and_prop

def get_v0(t,p):
    """
    simple linear velocity vector estimate, to allow obtaining a 
    good starting point guess
    p = ITRS position
    t = time
    """
    n_m=len(t)
    t_m = n.mean(t)
    t=t-t[0]
    A=n.zeros([n_m*3,6])
    A[0:n_m,0]=1.0
    A[0:n_m,3]=t
    A[n_m:(2*n_m),1]=1.0
    A[n_m:(2*n_m),4]=t
    A[(2*n_m):(3*n_m),2]=1.0
    A[(2*n_m):(3*n_m),5]=t
    m=n.concatenate((p[:,0],p[:,1],p[:,2]))
    xhat=n.linalg.lstsq(A,m)[0]
    p0=n.array([xhat[0],xhat[1],xhat[2]])
    v0=n.array([xhat[3],xhat[4],xhat[5]])    
    return(xhat,A,p0,v0)

def plot_linear_traj(ecef,t,xhat,A):
    n_m=len(t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecef[:,0],ecef[:,1],ecef[:,2],".")
    ax.plot([xhat[0]],[xhat[1]],[xhat[2]],"x")
    traj=n.dot(A,xhat)
    ax.plot(traj[0:n_m],traj[n_m:(2*n_m)],traj[(2*n_m):(3*n_m)])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecef[:,0]-traj[0:n_m],ecef[:,1]-traj[n_m:(2*n_m)],ecef[:,2]-traj[(2*n_m):(3*n_m)],".")
    plt.show()
    
def est_var(ecef,xhat,A):
    n_m=ecef.shape[0]
    traj=n.dot(A,xhat)
    return(n.array([n.var(ecef[:,0]-traj[0:n_m]),n.var(ecef[:,1]-traj[n_m:(2*n_m)]),n.var(ecef[:,2]-traj[(2*n_m):(3*n_m)])]))

def fit_model(t,p,v_est,p_est,ecef_std,hg,unix_t0,use_acc=True):
    """ 
    Nonlinear fit for trajectory 
    """
    t=t-t[0]
    def model(x):
        v0=10**x[0]
        u2=n.cos(x[1])
        u0=n.sin(x[1])*n.cos(x[2])
        u1=n.sin(x[1])*n.sin(x[2])
        if use_acc:
            v1=10**x[6]
            a1=10**x[7]
            a0=(v1-v0)/n.exp(a1*n.max(t))
            v = v0-a0*n.exp(a1*t)
        else:
            v = n.repeat(v0,len(t))
        modelf=n.zeros([len(t),3])
        dt=t[1]-t[0]
        # position offset to initial position measurement added
        # as unknown parameter
        modelf[0,:]=p_est + n.array([10.0*x[3],10.0*x[4],10.0*x[5]])
        for i in range(1,len(t)):
            modelf[i,0]=modelf[i-1,0]+u0*v[i]*(t[i]-t[i-1])
            modelf[i,1]=modelf[i-1,1]+u1*v[i]*(t[i]-t[i-1])
            modelf[i,2]=modelf[i-1,2]+u2*v[i]*(t[i]-t[i-1])
        return(modelf)
    
    def ss(x):
        m=model(x)
        s=0.0
        for i in range(3):
            s+=n.sum((1.0/(2.0*ecef_std[:,i]**2.0))*n.abs(m[:,i]-p[:,i])**2.0)
#            s+=n.sum(n.abs(m[:,i]-p[:,i])**2.0)            
        print(s)
        return(s)
    
    vn0=n.linalg.norm(v_est)
    u0=v_est/vn0
    # cos(theta)=z
    # sin(theta)*cos(phi)=x
    # sin(theta)*sin(phi)=y    
    theta=n.arccos(u0[2])
    phi=n.arccos(u0[0]/n.sin(theta))
    xhat=so.fmin(ss,[n.log10(vn0),theta,phi,0,0,0,n.log10(1.5*vn0),0])
    xhat=so.fmin(ss,xhat)
    xhat=so.fmin(ss,xhat)
    xhat=so.fmin(ss,xhat)        

    chain=scam.scam(ss,xhat,n_par=8,step=[0.001,0.001,0.001,0.5,0.5,0.5,0.002,0.002],n_iter=1000,thin=100,debug=False)
    mp=n.mean(chain,axis=0)
    
    plt.plot(chain[:,1])
    plt.plot(chain[:,2])
    plt.show()
    
    
    plt.plot(chain[:,0])
    plt.plot(chain[:,6])
#    plt.plot(chain[:,7])    
    plt.show()
    plt.plot(chain[:,3])
    plt.plot(chain[:,4])
    plt.plot(chain[:,5])    
    plt.show()

    m=model(mp)
    
    plt.plot(t,m[:,0]-p[:,0],".")
    plt.plot(t,m[:,1]-p[:,1],".")
    plt.plot(t,m[:,2]-p[:,2],".")
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Residual (m)")
    plt.tight_layout()
    plt.savefig("figs/traj_res.png")    
    plt.show()

    if use_acc:
        xv0=10**mp[0]
        xv1=10**mp[6]
        xa1=10**mp[7]
        
        xa0=(xv1-xv0)/n.exp(xa1*n.max(t))
        v = xv0-xa0*n.exp(xa1*t)
    else:
        v=n.repeat(10**mp[0],len(t))

    plt.plot(t,v/1e3)
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Velocity (km/s)")
    plt.tight_layout()    
    plt.savefig("figs/traj_vel.png")    
    plt.show()

    plt.plot(t[0:(len(t)-1)],n.diff(v)/n.diff(t)/1e3)
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Acceleration (km/s$^2$)")
    plt.tight_layout()    
    plt.savefig("figs/traj_acc.png")
    plt.show()

    vel_0s=[]
    vels=[]
    ras=[]
    decs=[]
    kep_states=[]
    for ci in range(chain.shape[0]):
        par=chain[ci,:]
        vel_0s.append(10**par[0])
        theta=par[1]
        phi=par[2]
        v0=10**par[0]
        z=n.cos(theta)
        x=n.sin(theta)*n.cos(phi)
        y=n.sin(theta)*n.sin(phi)
        vel=v0*n.array([x,y,z])
        vels.append(vel)
        u0=n.array([x,y,z])
        p0 = p_est + n.array([10.0*par[3],10.0*par[4],10.0*par[5]])
        rad=radiant_est.get_radiant(p0,unix_t0,u0).icrs
        ras.append(rad.ra.deg)
        decs.append(rad.dec.deg)
        
        state = n.zeros((6,), dtype=n.float64)
        state[:3] = p0
        state[3:] = vel
        kep_states.append(OD_and_prop.itrs_to_kep(state,unix_t0,dt=0.25))
    kep_states=n.array(kep_states)
        
    vels=n.array(vels)
    return(n.mean(vel_0s),n.std(vel_0s),
           n.mean(vels,axis=0),n.std(vels,axis=0),
           n.mean(ras),n.std(ras),
           n.mean(decs),n.std(decs),
           n.mean(kep_states,axis=0),n.std(kep_states,axis=0))



if __name__ == "__main__":
    # read trajectory determined using cameras
    h=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")

    # itrs pos
    ecef=n.copy(h[("ecef_pos_m")])
    ecef_std=n.copy(h[("ecef_pos_std")])
    # height
    hg=n.copy(h[("h_km_wgs84")])
    # unix time
    t=n.copy(h[("t_unix")])

#    plt.plot(hg,".")
 #   plt.show()
    
    # use a simple linear fit to get
    # initial velocity estimate
    xhat,A,p0,v0=get_v0(t,ecef)
    print(v0)
    # estimate radiant based on initial position and
    # velocity
    re0 = radiant_est.get_radiant(p0,t[0],v0/n.linalg.norm(v0))
    
    # Estimate variance of ITRS position measurements based on
    # residuals of initial fit
    #
    # It is important to make sure all measurements
    # are uncorrelated, or that their correlations
    # are correctly modeled!
#    sigma2=100.0*est_var(ecef,xhat,A)
 #   print("Stdev:")
  #  print(n.sqrt(sigma2))

    # non-linear fit with a first order atmospheric 
    # drag model.
    
    rv0,rv0s,v,vs,ra,ras,dec,decs,kep,kep_std=fit_model(t,ecef,v0,p0,ecef_std,hg,t[0])

    print("v0: %1.2f +/- %1.2f\nvx,vy,vz: %1.2f,%1.2f,%1.2f +/- %1.2f,%1.2f,%1.2f\nra,dec: %1.2f,%1.2f +/- %1.2f,%1.2f"%(rv0,rv0s,v[0],v[1],v[2],vs[0],vs[1],vs[2],ra,dec,ras,decs))

    print(re0.icrs.ra.deg)
    print(re0.icrs.dec.deg)
    
    
    ho=h5py.File("state_vector/v_ecef.h5","w")
    ho["v0"]=rv0
    ho["v0_sigma"]=rv0s
    ho["p0"]=p0
    ho["t0"]=t[0]
    ho["ra"]=ra
    ho["ra_sigma"]=ras    
    ho["dec"]=dec
    ho["dec_sigma"]=decs
    ho["vel"]=v
    ho["vel_sigma"]=vs
    ho["kep"]=kep
    ho["kep_std"]=kep_std
    ho.close()
    print(n.linalg.norm(v0))



#    v0: 27962.55 +/- 24.02
#vx,vy,vz: 22738.31,-8282.75,-14009.52 +/- 19.01,33.19,17.89
#ra,dec: 76.13,30.04 +/- 0.08,0.03
#76.20595750750215
#30.01801055221751
#27288.94268908626
#0: 27979.86 +/- 21.56
#vx,vy,vz: 22747.17,-8302.32,-14018.09 +/- 17.82,29.85,14.54
#ra,dec: 76.09,30.04 +/- 0.07,0.02
#76.20595750750215
#30.01801055221751
#27288.94268908626
#v0: 27977.84 +/- 30.63
#vx,vy,vz: 22747.06,-8299.35,-14016.08 +/- 20.22,47.06,17.61
#ra,dec: 76.10,30.04 +/- 0.10,0.03
#76.20595750750215
#30.01801055221751
#27288.94268908626
