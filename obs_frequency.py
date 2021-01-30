import numpy as n
import matplotlib.pyplot as plt

A_e=4.0*n.pi*6370e3**2.0
A_ski=n.pi*200e3**2.0

m=n.linspace(0.001,100.0,num=1000)

def mass2diam(m):
    return(2.0*((m/3000.0)*(3.0/4.0)*(1.0/n.pi))**(1/3.0))
def occurrence_rate(d):
    """ Brown et.al., 2002 """
    return(10**(1.568-2.7*n.log10(d)))


plt.loglog(m,occurrence_rate(mass2diam(m))*(A_ski/A_e))
plt.ylabel("Detections/year/station")
plt.xlabel("Mass (kg)")
plt.show()
