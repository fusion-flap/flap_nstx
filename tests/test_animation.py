import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.animation as animation

def f(x,y,a):
    return a*(x**2+y**2)

avals = list(np.linspace(0,1,10))
xaxis = list(np.linspace(-2,2,9))
yaxis = list(np.linspace(-2,2,9))

xy = list(itertools.product(xaxis,yaxis))
xy = np.array(list(map(list,xy)))

x = xy[:,0]
y = xy[:,1]

zlist = []

for a in avals:
    z = []
    for i, xval in enumerate(x):
        z.append(f(x[i],y[i],a))
    zlist.append(z)

xi = np.linspace(min(x),max(x),len(x))
yi = np.linspace(min(y), max(y), len(y))

zmin = min([min(zl) for zl in zlist])
zmax = max([max(zl) for zl in zlist])
levels = np.linspace(zmin, zmax,41)
kw = dict(levels=levels, cmap=plt.cm.hsv, vmin=zmin, vmax=zmax, origin='lower')

fig,ax = plt.subplots()
zi = ml.griddata(x, y, zlist[0], xi, yi, interp='linear')
contourplot = ax.contourf(xi, yi, zi, **kw)
cbar = plt.colorbar(contourplot)

def animate(index):
    zi = ml.griddata(x, y, zlist[index], xi, yi, interp='linear')
    ax.clear()
    ax.contourf(xi, yi, zi, **kw)
    ax.set_title('%03d'%(index))


ani = animation.FuncAnimation(fig,animate,10,interval=200,blit=False)
plt.show()