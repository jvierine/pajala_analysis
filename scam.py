import numpy as n

def scam(ss,x0,n_par=2,step=[0.01,0.01],n_iter=1000,thin=10,debug=False):
    chain=n.zeros([n_iter,n_par],dtype=n.float32)
    for i in range(n_iter*thin):
        lp = ss(x0)
        xtry=n.copy(x0)
        pi = int(n.random.rand(1)*n_par)
        xtry[pi]+=n.random.randn(1)*step[pi]
        lp_try = ss(xtry)
        if lp_try <= lp:
            x0=xtry
        else:
            alpha=n.log(n.random.rand(1))
            if alpha < (lp - lp_try):
                x0=xtry

        chain[int(i/thin),:]=x0
        if debug:
            print(lp)
    return(chain)
