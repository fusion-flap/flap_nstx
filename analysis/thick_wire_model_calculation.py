#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 23:48:01 2020

@author: mlampert
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#func = lambda x0,x1,x2,x3,d : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (1 if (x0-.2*x3-.5-.25*x1>0) else 0)
#
#points = [[lambda x1,x2,x3 : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]
#def opts0(*args, **kwargs):
#    return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
#d=1.
#a=integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
#                opts=[opts0,{},{},{}],args=(1))
#print(a)
def thick_wire_estimation():
    func = lambda r1,r2,theta1,theta2,d,d2: r1*r2*(r2*np.cos(theta2)-r1*np.cos(theta1)+d)/((r2*np.cos(theta2)-r1*np.cos(theta1)+d)**2+(r2*np.sin(theta2)-r1*np.sin(theta1))**2)
    
    d=np.arange(10)/100+0.01
    
    integral=np.zeros(10)
    
    for i in range(len(d)):
    
        integral[i]=integrate.nquad(func, [[0,0.01], [0,0.01], [0,2*np.pi], [0,2*np.pi]], args=(d[i],d[i]))
        print(integral[i])
    plt.figure()
    plt.plot(d,integral)
    
def thick_wire_estimation_numerical(j0=0.4e6,
                                    r=0.01,
                                    d=0.02,
                                    n_mesh=30,
                                    n_i=5e19,
                                    acceleration=True,
                                    test=False,
                                    ):
    m_i=2.014*1.66e-27
    
    def integrable(r1,r2,theta1,theta2,d):
        r1costheta1=r1*np.cos(theta1)
        r2costheta2=r2*np.cos(theta2)
        
        return r1*r2*(r2costheta2-r1costheta1+d)/((r2costheta2-r1costheta1+d)**2+(r2*np.sin(theta2)-r1*np.sin(theta1))**2)
    
    r_mesh=np.arange(n_mesh)/(n_mesh-1)*r
    theta_mesh=[]
    for i in range(2,n_mesh+2):
        #theta_mesh.append(np.arange(int(n_mesh*np.sqrt(i/n_mesh)))/int(n_mesh*np.sqrt(i/n_mesh))*np.pi*2)
        theta_mesh.append(np.arange(i)/(i-1)*np.pi*2)
        
            
    if test:
        plt.figure()
        for i in range(n_mesh):
            r=np.zeros(theta_mesh[i].shape[0]) 
            r[:]=r_mesh[i]
            plt.plot(r*np.cos(theta_mesh[i]),r*np.sin(theta_mesh[i]))
            
    #print(theta_mesh)
    integral=0.
    
    for i in range(n_mesh-1):
        for j in range(n_mesh-1):
            for k in range(len(theta_mesh[i])-1):
                for l in range(len(theta_mesh[j])-1):
                    for perm1 in [0,1]:
                        for perm2 in [0,1]:
                            for perm3 in [0,1]:
                                for perm4 in [0,1]:
                                    integral+=integrable(r_mesh[i+perm3],
                                                         r_mesh[j+perm4],
                                                         theta_mesh[i][k+perm1],
                                                         theta_mesh[j][l+perm2],
                                                         d)/16.*(r_mesh[i]-r_mesh[i+1])*(r_mesh[j]-r_mesh[j+1])*(theta_mesh[i][k]-theta_mesh[i][k+1])*(theta_mesh[j][l]-theta_mesh[j][l+1])
    
    
#    print(integral)
    if not acceleration:
        return j0**2 * 2e-7 * integral
    else:
        return j0**2 * 2e-7 * integral/(n_i*r**2 * 2*np.pi*m_i)


def test_thick_wire_estimation_numerical(d1=0.001,
                                         d2=0.1,
                                         nd=24):
    m_i=2.014*1.66e-27
    n_i=0.5e19
    d=np.arange(nd)/nd*(d2-d1)+d1
    force=np.zeros(nd)
    for i in range(nd):
        force[i]=thick_wire_estimation_numerical(r=0.013,
                                                d=d[i],
                                                n_mesh=30,
                                                test=False,
                                                acceleration=True,
                                                )
    plt.figure()
    plt.plot(d,force)
    print(d,force)
    acceleration=force/(m_i*n_i*0.01**2*np.pi)
    
    return acceleration

def test_thick_wire_againts_thin_wire(d1=0.001,
                                      d2=0.1,
                                      n_mesh=30,
                                      nd=12):
    m_i=2.014*1.66e-27
    n_i=0.5e19
    r=0.013
    j0=0.4e6
    
    d=np.arange(nd)/nd*(d2-d1)+d1
    force_thick=np.zeros(nd)
    force_thin=np.zeros(nd)
    
    for i in range(nd):
        force_thick[i]=thick_wire_estimation_numerical(j0=j0,
                                                        r=r,
                                                        d=d[i],
                                                        n_mesh=n_mesh,
                                                        test=False,
                                                        acceleration=False,
                                                        )
        
        force_thin[i]=4*np.pi*1e-7 * (j0 * r**2 * np.pi)**2 / (2*np.pi*d[i])
        
    plt.figure()
    plt.plot(d/r,force_thick, color='blue')
    plt.plot(d/r,force_thin, color='red')
    plt.xlabel('d/r')
    plt.ylabel('Force [N]')
    plt.title('d/r vs Force for thin (red) and thick (blue) wires.')
    print(force_thick/force_thin)
    acceleration=force_thick/(m_i*n_i*0.01**2*np.pi)
    return force_thick,force_thin