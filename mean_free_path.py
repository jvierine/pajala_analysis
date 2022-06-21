import numpy as n
import matplotlib.pyplot as plt
import scipy.optimize as sio
import scipy.interpolate as si

def mfp(h):
    """
    mean free path (in meters) as a function of height h (km)
    """
    pc=n.polyfit([90,115],[-1.6989,0.2788],1)

    pf=n.poly1d(pc)
    return(10**pf(h))
    


if __name__ == "__main__":

    h=n.linspace(90,115,num=100)
    plt.semilogx(mfp(h),h)
    plt.xlabel("Mean free path (m)")
    plt.ylabel("Height (km)")
    plt.grid()
    plt.show()
