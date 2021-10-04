#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:07:27 2021

@author: mlampert
"""

import numpy as np
from numpy.linalg import eig, inv

class FitEllipse:
    """
    Wrapper class for fitting an Ellipse and returning its important features.
    It uses the least square approximation method combined with a Lagrangian
    minimalization for the Eigenvalues of the problem.
    Source:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
        https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645
        Fitzgibbon, Pilu and Fischer in Fitzgibbon, A.W., Pilu, M., and Fischer R.B., Direct least squares fitting of ellipsees, 
        Proc. of the 13th Internation Conference on Pattern Recognition, pp 253–257, Vienna, 1996
    Rewritten as an object.
    """
    def __init__(self,  
                 x=None,                                                        #The x coordinates of the input data as a numpy array
                 y=None,                                                        #The y coordinates of the input data as a numpy array
                 ):
        """
        Initializes (fits) the ellipse onto the given x and y data points
        """
        if x is None or y is None:
            raise TypeError('x or y is not set.')
        if len(x) < 6 or len(y) < 6:
            raise ValueError('There should be 6 points defining the ellipse.')
            
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        self.x=x
        self.y=y
        self.parameters = V[:,n]
        
    @property
    def center(self):
        """
        Returns the center of the ellipse.
        """
        p=self.parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return np.array([x0,y0])
    
    @property
    def axes_length(self):
        """
        Returns the minor and major axes lengths of the ellipse.
        """
        p=self.parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1=(b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    @property
    def angle_of_rotation(self):
        """
        Returns the angle of rotation compared to horizontal.
        """
        p=self.parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        if b == 0:
            return 0.
        else:
            return np.arctan(2*b/(a-c))/2
        #There is a modification online which makes the whole fitting fail due to
        #a wrong angle of rotation.
#        if b == 0:
#            if a > c:
#                return 0
#            else:
#                return np.pi/2
#        else: 
#            if a > c:
#                return np.arctan(2*b/(a-c))/2
#            else:
#                return np.pi/2 + np.arctan(2*b/(a-c))/2
    @property
    def size(self):
        """
        Returns the size of the ellipse in the x and y direction along its center.
        Not part of the description in the source.
        a contains the coefficients like this:
        a[0]*x**2 + a[1]*x*y + a[2]*y**2 + a[3]*x + a[4]*y + a[5] = 0
        The sizes are calculated as the solution of the 2nd order equation:
        ysize=np.abs(y1-y2) where y1,y2 = y1,2(x=x0)
        xsize=np.abs(x1-x2) where x1,x2 = x1,2(y=y0)
        """
        a=self.parameters
        x0,y0=self.center
        
        ax=a[0]
        ay=a[2]
        bx=a[1]*y0+a[3]
        by=a[1]*x0+a[4]
        cx=a[2]*y0**2+a[4]*y0+a[5]
        cy=a[0]*x0**2+a[3]*x0+a[5]
        xsize=np.sqrt(bx**2-4*ax*cx)/np.abs(ax)
        ysize=np.sqrt(by**2-4*ay*cy)/np.abs(ay)
        if np.imag(xsize) != 0 or np.imag(xsize) !=0:
            print('size is complex')
            raise ValueError('')
        return np.array([xsize,ysize])