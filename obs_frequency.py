import numpy as n
import matplotlib.pyplot as plt

A_e=4.0*n.pi*6370e3**2.0

# collecting area of Skibotn
A_ski=n.pi*250e3**2.0

m=n.linspace(0.001,1000.0,num=1000)

def mass2diam(m):
    return(2.0*((m/3000.0)*(3.0/4.0)*(1.0/n.pi))**(1/3.0))
def occurrence_rate(d):
    """ Brown et.al., 2002 """
    return(10**(1.568-2.7*n.log10(d)))

if __name__ == "__main__":
    
    print(occurrence_rate(mass2diam(12))*(A_ski/A_e))
    print(occurrence_rate(mass2diam(170))*(A_ski/A_e))
    
    print(occurrence_rate(mass2diam(12)))
    print(occurrence_rate(mass2diam(170)))
    
    plt.loglog(m,occurrence_rate(mass2diam(m))*(A_ski/A_e))
    plt.ylabel("Detections/year/station")
    plt.xlabel("Mass (kg)")
    plt.show()
