import h5py
import numpy as n
import matplotlib.pyplot as plt
import stuffr
import scipy.optimize as so

def time_freq():
    d=n.array([
        [848.8295567170398, 15.7952715905633],
        [901.436231362849, 15.181768032872853],
        [989.1140224391984, 14.629614830951454],
        [1032.9529179773735, 14.261512696337189],
        [1024.1851388697387, 13.525308427108655],
        [1024.1851388697387, 13.116306055315025],
        [1068.024034407913, 12.850454513649165],
        [1111.8629299460872, 12.461902260445218],
        [1138.1662672689927, 12.175600600189679],
        [1173.237383699532, 11.623447398268276],
        [1208.3085001300715, 11.337145738012737],
        [1208.3085001300715, 11.21444502647465],
        [1252.1473956682466, 10.866793010450063],
        [1260.9151747758815, 10.744092298911973],
        [1304.7540703140558, 10.416890401477069],
        [1339.825186744596, 10.151038859811212],
        [1366.1285240675006, 9.946537673914396],
        [1392.4318613904052, 9.414834590582679],
        [1401.19964049804, 9.251233641865227],
        [1427.5029778209446, 8.883131507250958],
        [1427.5029778209446, 8.739980677123189],
        [1462.574094251485, 8.351428423919241],
        [1488.8774315743894, 8.167377356612107],
        [1515.180768897294, 8.04467664507402],
        [1532.7163271125637, 7.9015258149462495],
        [1567.787443543104, 7.697024629049435],
        [1602.8585599736434, 7.410722968793894],
        [1637.9296764041828, 7.1857716643073966],
        [1664.2330137270883, 6.960820359820901],
        [1690.5363510499928, 6.735869055334404],
        [1734.375246588167, 6.429117276489182],
        [1751.9108048034377, 6.24506620918205],
        [1813.2852585568817, 5.897414193157464],
        [1874.6597123103265, 5.590662414312242],
        [1892.1952705255962, 5.529312058543197]])

    d2 = n.array([
        [342,13],
        [402,12.9],
        [460,12.1],
        [519,11.5],
        [578,11.2],                
    ])


    logtau=n.log10(d[:,0])
    logf = n.log10(d[:,1])
    nm=len(d[:,0])
    A=n.zeros([nm,2])
    A[:,0]=1.0
    A[:,1]=logtau

    xhat=n.linalg.lstsq(A,logf)[0]
    print(xhat)
    def model(x):
        return(10**(x[0]+x[1]*logtau))

    errvar=n.mean(n.abs(n.dot(A,xhat)-logf)**2.0)
#    print(errvar)
    print("error variance")
    print(n.sqrt(n.diag(errvar*n.linalg.inv(n.dot(A.T,A)))))
    
    plt.loglog(d[:,0],model(xhat))
    plt.loglog(d[:,0],d[:,1],".")
    plt.show()
    print(xhat)

    
        
    
    return(d)


tfd=time_freq()

h=h5py.File("oblique_data/sod_to_ski.h5","r")

print(h.keys())

rgs=h["ranges"][()]
times=h["unix_time"][()]
snrs=h["SNR"][()]
freqs=h["freqs"][()]

idx=n.where( (rgs > 420) & (rgs < 460) ) [0]
fig = plt.figure()


t0=stuffr.date2unix(2020,12,4,13,30,38)
print(len(rgs))
print(len(freqs))
print(len(times))
S=n.max(snrs[:,idx,:],axis=1)
plt.pcolormesh(times-t0,freqs,10.0*n.log10(S.T),vmin=0,vmax=20)
plt.plot(tfd[:,0],tfd[:,1],"+",color="white",alpha=0.8)
plt.ylim([1,16])
plt.xlim([-500,6000])
#plt.xscale("log")
#plt.yscale("log")
plt.axvline(0,color="red")
cb=plt.colorbar()
cb.set_label("SNR (dB)")
plt.title("Sodankylä-Skibotn oblique path")
plt.xlabel("Time after atmospheric entry (s)")
plt.ylabel("Frequency (MHz)")
plt.tight_layout()

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix,iy)

    global coords
    coords = [ix, iy]

    return coords


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.savefig("figs/paj_iono_e_snr.png")
plt.show()
print(snrs.shape)
#snrs[]
