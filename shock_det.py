import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob
import sys
import h5py
import imageio
import glob
import os
import scipy.interpolate as si
import scipy.io as sio
import stuffr
import station_coords as sc
import jcoord
ts=[]
xs=[]
ys=[]
tnow=0.0
# 
#ski
#odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/ski_cam/frames",
#                  t0=1607088628.0 - 13.0/15.0
#                   i0=151,
#odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/ski_cam/frames_30"
#t0=1607088643.0+1.0/15.0
#i0=0
#
# sor
# t0=1607088637.0 - 9.0/15.0
#
def extract_frames(odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/ski_cam/frames",
                   t0=1607088628.0 - 13.0/15.0,
                   i0=151,
                   dtau=0.21,
                   dt=1.0/15.0):
    
    global ts,xs,ys,tnow
    fl=glob.glob("%s/*.jpg"%(odir))
    fl.sort()
    frame_num=0.0
    h1=h5py.File("%s/clicks2.h5"%(odir),"r")
    ski_t=n.copy(h1["t"].value)
    ski_t[0]=ski_t[0]-3600
    ski_t[-1]=ski_t[-1]+3600    
    ski_xs=h1["xs"][()]
    ski_ys=h1["ys"][()]
    ski_xf=si.interp1d(ski_t,ski_xs)
    ski_yf=si.interp1d(ski_t,ski_ys)
    
    
    ht=h5py.File("camera_data/2020-12-04-trajectory_std.h5","r")
    t_pos=n.copy(ht["ecef_pos_m"][()])
    t_t=n.copy(ht["t_unix"][()])
    t_t[0]=t_t[0]-3600
    t_t[-1]=t_t[-1]+3600    
    ht.close()
    tx=si.interp1d(t_t,t_pos[:,0])
    ty=si.interp1d(t_t,t_pos[:,1])
    tz=si.interp1d(t_t,t_pos[:,2])    
    print(t_pos.shape)

    ski_p = jcoord.geodetic2ecef(sc.ski_coords[0],sc.ski_coords[1],sc.ski_coords[2])

    cal=sio.loadmat("camera_data/azzeSorSki.mat")
            
    for fi in range(len(fl)):
        if fi > i0:
            tnow = fi*dt + t0 - dtau
            fname="%s/frame-%06d.h5"%(odir,fi)
            print(fname)
            h=h5py.File(fname,"r")
            I=n.copy(h["I"].value)
            fig, ax = plt.subplots(figsize=(14,10))
            bg=n.median(I)            


            R=n.linalg.norm(ski_p-n.array([tx(tnow),ty(tnow),tz(tnow)]))

            idxm=200
            idxp=100
            idym=80
            idyp=80

            # center pos
            zes_ski = 180.0*cal["zeSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            azs_ski = 180.0*cal["azSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            llh=jcoord.az_el_r2geodetic(sc.ski_coords[0], sc.ski_coords[1],sc.ski_coords[2], azs_ski, 90-zes_ski, R)
            p_center=jcoord.geodetic2ecef(llh[0],llh[1],llh[2])
            
            # left pos
            zes_ski = 180.0*cal["zeSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)-idxm))]/n.pi
            azs_ski = 180.0*cal["azSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)-idxm))]/n.pi
            llh=jcoord.az_el_r2geodetic(sc.ski_coords[0], sc.ski_coords[1],sc.ski_coords[2], azs_ski, 90-zes_ski, R)
            p_left=jcoord.geodetic2ecef(llh[0],llh[1],llh[2])

            # right pos
            zes_ski = 180.0*cal["zeSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)+idxp))]/n.pi
            azs_ski = 180.0*cal["azSki"][int(n.round(ski_yf(tnow+dtau))),int(n.round(ski_xf(tnow+dtau)+idxp))]/n.pi
            llh=jcoord.az_el_r2geodetic(sc.ski_coords[0], sc.ski_coords[1],sc.ski_coords[2], azs_ski, 90-zes_ski, R)
            p_right=jcoord.geodetic2ecef(llh[0],llh[1],llh[2])

            # top pos
            zes_ski = 180.0*cal["zeSki"][int(n.round(ski_yf(tnow+dtau)+idyp)),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            azs_ski = 180.0*cal["azSki"][int(n.round(ski_yf(tnow+dtau)+idyp)),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            llh=jcoord.az_el_r2geodetic(sc.ski_coords[0], sc.ski_coords[1],sc.ski_coords[2], azs_ski, 90-zes_ski, R)
            p_top=jcoord.geodetic2ecef(llh[0],llh[1],llh[2])

            # bottom pos
            zes_ski = 180.0*cal["zeSki"][int(n.round(ski_yf(tnow+dtau)-idym)),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            azs_ski = 180.0*cal["azSki"][int(n.round(ski_yf(tnow+dtau)-idym)),int(n.round(ski_xf(tnow+dtau)))]/n.pi
            llh=jcoord.az_el_r2geodetic(sc.ski_coords[0], sc.ski_coords[1],sc.ski_coords[2], azs_ski, 90-zes_ski, R)
            p_bottom=jcoord.geodetic2ecef(llh[0],llh[1],llh[2])
            
            left_d=n.linalg.norm(p_center-p_left)
            right_d=n.linalg.norm(p_center-p_right)
            
            top_d=n.linalg.norm(p_center-p_top)
            bottom_d=n.linalg.norm(p_center-p_bottom)            

            
            plt.imshow(I[int(ski_yf(tnow+dtau)-idym):int(ski_yf(tnow+dtau)+idyp),int(ski_xf(tnow+dtau)-idxm):int(ski_xf(tnow+dtau)+idxp)],cmap="gray",vmin=bg,vmax=400,aspect="auto",extent=[-left_d,right_d,-bottom_d,top_d])            
            plt.title("%s R=%1.2f km"%(stuffr.unix2datestr(tnow),R/1e3))
            plt.colorbar()
            plt.xlabel("Distance (m)")
            plt.ylabel("Distance (m)")     
#            plt.xlim([ski_xf(tnow)-300,ski_xf(tnow)+100])
 #           plt.ylim([ski_yf(tnow)-70,ski_yf(tnow)+70])            
            plt.show()


if __name__ == "__main__":
    # skibotn
    extract_frames(odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/ski_cam/frames",
                   t0=1607088628.0 - 13.0/15.0,
                   i0=147,
                   dt=1.0/15.0)
