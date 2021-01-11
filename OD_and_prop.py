#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
import h5py
import scipy.constants as c
import sorts
import pyorb
import numpy as n


import rebound
    
def sample_kep(state_file = 'state_vector/v_ecef.h5',
               dt=0.25,
               n_samples=100):

    h=h5py.File(state_file, 'r')
    state = np.zeros((6,), dtype=np.float64)

    kep_states=[]
    for i in range(n_samples):
        state[:3] = h['p0'][()]
        state[3:] = h['vel'][()]
        t0=n.copy(h['t0'][()])
        epoch = Time(t0+n.random.randn(1)[0]*dt, scale='utc', format='unix')
        state[3:] += n.random.randn(3)*h["vel_sigma"][()]

        state_helio = sorts.frames.convert(epoch, state, in_frame='ITRS', out_frame='heliocentricmeanecliptic')
        state_helio.shape = (6,1)
        
        orb = pyorb.Orbit(
            M0 = pyorb.M_sol,
            direct_update=True,
            auto_update=True,
            degrees = True,
        )
        orb.cartesian = state_helio
        tis=(5.2044/(orb.a/c.au)) + 2.0*n.cos(n.pi*orb.i/180.0)*n.sqrt( ((orb.a/c.au)/5.2044)*(1-orb.e**2.0))
        kep_state=[orb.a,orb.e,orb.i,orb.omega,orb.Omega,orb.anom,tis]
        print(kep_state)
        kep_states.append(kep_state)
    kep_states=n.array(kep_states)
    print(n.mean(kep_states,axis=0))
    print(n.std(kep_states,axis=0))
    print(orb)


def itrs_to_kep(state,t0,dt=0.0):
    #    state = np.zeros((6,), dtype=np.float64)
    #        state[:3] = h['p0'][()]
    #       state[3:] = h['vel'][()]
    #        t0=n.copy(h['t0'][()])
    epoch = Time(t0+n.random.randn(1)[0]*dt, scale='utc', format='unix')
    #    state[3:] += n.random.randn(3)*h["vel_sigma"][()]
    
    state_helio = sorts.frames.convert(epoch, state, in_frame='ITRS', out_frame='heliocentricmeanecliptic')
    state_helio.shape = (6,1)
        
    orb = pyorb.Orbit(
        M0 = pyorb.M_sol,
        direct_update=True,
        auto_update=True,
        degrees = True,
    )
    orb.cartesian = state_helio
    tis=(5.2044/(orb.a/c.au)) + 2.0*n.cos(n.pi*orb.i/180.0)*n.sqrt( ((orb.a/c.au)/5.2044)*(1-orb.e**2.0))
    kep_state=[orb.a,orb.e,orb.i,orb.omega,orb.Omega,orb.anom,tis]
    return(kep_state,state_helio)

    
def rebound_stuff():
    """
    "Zenith attraction"
    """
    #        print('Initial orbit:')
    #       print(orb)
    #      print("Perhelion distance %1.2f (AU)"%(orb.a*(1-orb.e)/c.au))
    ho=h5py.File("state_vector/v_ecef.h5","r")
    v0=n.copy(ho["vel"][()])
    p0=n.copy(ho["p0"][()])
    t0=n.copy(ho["t0"][()])
    ho.close()
    state = np.zeros((6,), dtype=np.float64)    
    state[:3] = p0
    state[3:] = v0
    print(state)

    earth_idx=4
    jup_idx=6
    
    kep_state,state_helio = itrs_to_kep(state,t0)
    dt = 1.0
    t_long = np.linspace(-30*365.25*24*3600,0,num=10000)
    t0_max = 5*24*3600.0
    n_t=int(2*t0_max/dt)
    t_short = np.linspace(-t0_max,t0_max,num=n_t)
    t=np.concatenate((t_short,t_long))
    t=np.unique(t)
    t=np.sort(t)
#    print("dt")
 #   print(dt)
    

    epoch=Time(t0,scale="utc",format="unix")
    print(epoch)



    #This meta kernel lists de430.bsp and de431 kernels as well as others like the leapsecond kernel.
    spice_meta = 'spice_kernels0.txt'
    
    #We input in International Terrestrial Reference System coordinates
    #and output in International Celestial Reference System
    #going trough HeliocentricMeanEclipticJ2000 internally in Rebound
    prop = sorts.propagator.Rebound(
        spice_meta = spice_meta, 
        settings=dict(
            in_frame='HeliocentricMeanEcliptic',
            out_frame='HeliocentricMeanEcliptic',
            time_step = dt, #s
    ),
    )
    prop.orb_el=True
    states,mob_states = prop.propagate([-365.24*24*3600,-2*31.0*24*3600,-31.0*24*3600,-24*3600,-3600,-60,0], state_helio, epoch)
    print(state_helio)
    print(states)
#    exit(0)
    prop.orb_el=False

    print("propagating")
    states,mob_states = prop.propagate(t, state_helio, epoch)
    print("done")
    cmp_idx=earth_idx
    plt.plot(t,n.sqrt( (mob_states[0,:,cmp_idx]-states[0,:])**2.0+
                       (mob_states[1,:,cmp_idx]-states[1,:])**2.0+
                       (mob_states[2,:,cmp_idx]-states[2,:])**2.0 )/1e3)
    plt.xlabel("Time (s)")
    plt.ylabel("Earth distance (km)")
    plt.show()
    
    #plot results
    plot3d=False
    states = states-mob_states[:,:,0]
    for i in range(mob_states.shape[2]):
        mob_states[:,:,i] = mob_states[:,:,i]-mob_states[:,:,0]
    if plot3d:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[0,:]-mob_states[0,:,0], states[1,:], states[2,:], "-b")
        for i in range(mob_states.shape[2]):
            ax.plot(mob_states[0,:,i], mob_states[1,:,i], mob_states[2,:,i])
        plt.xlabel("X")
        plt.ylabel("Y")
    else:
        mob_labels=["Mercury","Venus","Earth","Mars","Jupiter"]
        mob_idx=[2,3,4,5,6]
        bidx=n.where(t < 0)[0]
        aidx=n.where(t > 0)[0]
        plt.plot([0],[0],"o",color="yellow",label="Sun")        
        plt.plot(states[0,bidx],states[1,bidx],color="blue",label="Meteor")
        plt.plot(states[0,aidx],states[1,aidx],color="red")        
        for ii,i in enumerate(mob_idx):
            plt.plot(mob_states[0,:,i], mob_states[1,:,i],label=mob_labels[ii])
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Heliocentric mean ecliptic J2000")
        plt.legend()
        plt.axes().set_aspect('equal')
        plt.show()

    #    plt.zlabel("Z")
    
    plt.show()

def rebound_stuff2():
    
    ho=h5py.File("state_vector/v_ecef.h5","r")
    v0=n.copy(ho["vel"][()])
    p0=n.copy(ho["p0"][()])
    t0=n.copy(ho["t0"][()])
    ho.close()
    state = np.zeros((6,), dtype=np.float64)

    state[:3] = p0
    state[3:] = v0
    print(state)
    print(state)
    
    kep_state,state_helio = itrs_to_kep(state,t0)
    print(state_helio)
    epoch=Time(t0,scale="utc",format="unix")
    print(epoch)

    state_helio=state_helio.flatten()
    sim = rebound.Simulation()
    sim.units=("s","m","kg")
    sim.add("Sun")
    sim.add("Earth")    
    sim.add("Jupiter")
    sim.add("Saturn")
    sim.status()
    print(sim.units)    
    sim.add(m=10.0,x=state_helio[0],y=state_helio[1],z=state_helio[2],
            vx=state_helio[3],vy=state_helio[4],vz=state_helio[5])
            
    for orbit in sim.calculate_orbits():
        print(orbit)

    #sim.add("Churyumov-Gerasimenko")
#    sim.add("NAME=Churyumov-Gerasimenko; CAP")
    fig = rebound.OrbitPlot(sim, unitlabel="[m]")
    plt.show()
rebound_stuff()    
#rebound_stuff2()
