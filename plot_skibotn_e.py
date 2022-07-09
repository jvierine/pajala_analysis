import h5py
import numpy as n
import matplotlib.pyplot as plt



h=h5py.File("oblique_data/sod_to_ski.h5","r")

print(h.keys())

rgs=h["ranges"][()]
times=h["unix_time"][()]
snrs=h["SNR"][()]
freqs=h["freqs"][()]

idx=n.where( (rgs > 420) & (rgs < 460) ) [0]

print(snrs.shape)
#snrs[]
