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


def itrs_to_kep(state,t0,dt=0.25):
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
    print(kep_state)
    return(kep_state)
#    kep_states.append(kep_state)
 #   kep_states=n.array(kep_states)
  #  print(n.mean(kep_states,axis=0))
   # print(n.std(kep_states,axis=0))
    #print(orb)

    
def rebound_stuff():
    #        print('Initial orbit:')
    #       print(orb)
    #      print("Perhelion distance %1.2f (AU)"%(orb.a*(1-orb.e)/c.au))
    
    sample_kep()
    dt = 3600.0
    t_end = 100*365.25*24*3600.0
    t = -np.arange(0, t_end, dt)
    

    #This meta kernel lists de430.bsp and de431 kernels as well as others like the leapsecond kernel.
    spice_meta = '/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/MetaK.txt'
    
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
    
    bstates = prop.propagate(t, state_helio, epoch)
    
    
    #plot results
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[0,:], states[1,:], states[2,:], "-b")
    
    plt.show()
