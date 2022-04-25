import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint

def rhs(u,t):
    y1,y2,y3,y4 = u
    a=0.012277471; b=1.0-a;
    D1=((y1+a)**2+y2**2)**(3.0/2);
    D2=((y1-b)**2+y2**2)**(3.0/2);
    res = [y3,\
           y4,\
           y1+2.0*y4-b*(y1+a)/D1-a*(y1-b)/D2, \
           y2-2.0*y3-b*y2/D1-a*y2/D2
           ]

    return res

t=np.linspace(0,20,1000)
res=odeint(rhs,[0.994,0.0,0.0,-2.0317326295573368357302057924],t)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
y1r,y2r,y3r,y4r=res[:, 0],res[:, 1],res[:, 2],res[:, 3]
ax.plot(y1r,y2r ,color="darkblue")
ax.scatter(0.1,0.1,color="green",marker="o", s=100, label='Earth')
ax.scatter (1.0,0.1,color="grey", marker="o", s=100, label ='Moon')
ax.legend(loc="upper left", fontsize=14)
plt.show()
