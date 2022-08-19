#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:14:28 2022

@author: mlampert
"""
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n =  np.argmax(E)
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes

if __name__ == '__main__':

    points = [( 0 , 3),
        ( 1 , 2),
        ( 1 , 7),
        ( 2 , 2),
        ( 2 , 4),
        ( 2 , 5),
        ( 2 , 6),
        ( 2 ,14),
        ( 3 , 4),
        ( 4 , 4),
        ( 5 , 5),
        ( 5 ,14),
        ( 6 , 4),
        ( 7 , 3),
        ( 7 , 7),
        ( 8 ,10),
        ( 9 , 1),
        ( 9 , 8),
        ( 9 , 9),
        (10,  1),
        (10,  2),
        (10 ,12),
        (11 , 0),
        (11 , 7),
        (12 , 7),
        (12 ,11),
        (12 ,12),
        (13 , 6),
        (13 , 8),
        (13 ,12),
        (14 , 4),
        (14 , 5),
        (14 ,10),
        (14 ,13)]

    fig, axs = plt.subplots(2, 1, sharex = True, sharey = True)
    a_points = np.array(points)
    x = a_points[:, 0]
    y = a_points[:, 1]
    axs[0].scatter(x,y)
    center, phi, axes = find_ellipse(x, y)
    print("center = ",  center)
    print("angle of rotation = ",  phi)
    print("axes = ", axes)

    axs[1].scatter(x, y)
    axs[1].scatter(center[0],center[1], color = 'red', s = 100)
    axs[1].set_xlim(x.min(), x.max())
    axs[1].set_ylim(y.min(), y.max())

    ell_patch = Ellipse(center, 2*axes[0], 2*axes[1], phi, edgecolor='red', facecolor='none')
    
    axs[1].add_patch(ell_patch)

    plt.show()
    
    from skimage.measure import EllipseModel