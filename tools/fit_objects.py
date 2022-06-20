#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:07:27 2021

@author: mlampert
"""

import numpy as np
from numpy.linalg import eig, inv
import scipy
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt

class FitEllipse:
    
    def __init__(self,  
                 x=None,                                                        #The x coordinates of the input data as a numpy array
                 y=None,                                                        #The y coordinates of the input data as a numpy array
                 method='linalg'                                                #Linalg or skimage
                 ):
        if method not in ['linalg','skimage']:
            raise ValueError('method needs to be either linalg or skimage!')
            
        """
        Initializes (fits) the ellipse onto the given x and y data points
        """
        if x is None or y is None:
            raise TypeError('x or y is not set.')
        if len(x) != len(y):
            raise ValueError('The length of x and y should be the same.')
            
        if len(x) < 6 or len(y) < 6:
            x_new=np.zeros(len(x)*2)
            y_new=np.zeros(len(x)*2)
            for i in range(len(x)):
                x_new[2*i]=x[i]
                y_new[2*i]=y[i]
                if i != len(x)-1:
                    x_new[2*i+1]=(x[i]+x[i+1])/2
                    y_new[2*i+1]=(y[i]+y[i+1])/2
                else:
                    x_new[2*i+1]=(x[i]+x[0])/2
                    y_new[2*i+1]=(y[i]+y[0])/2
            x=x_new
            y=y_new
            
            #raise ValueError('There should be at least 6 points defining the ellipse.')
        self.xmean=np.mean(x)
        self.ymean=np.mean(y)
        
        try:
            if method=='linalg':
                self._fit_ellipse(x, y)
            elif method=='skimage':
                self._fit_ellipse_skimage(x, y)
        except:
            self.set_invalid()

    def set_invalid(self):
        
        self._angle=np.nan
        self._axes_length=np.asarray([np.nan,np.nan])
        self._center=np.asarray([np.nan,np.nan])
        self._parameters=np.asarray([np.nan]*6)
        
    def _fit_ellipse(self,x,y):
        """
        Wrapper class for fitting an Ellipse and returning its important features.
        It uses the least square approximation method combined with a Lagrangian
        minimalization for the Eigenvalues of the problem.
        Source:
            https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645
            Fitzgibbon, Pilu and Fischer in Fitzgibbon, A.W., Pilu, M., and Fischer R.B., Direct least squares fitting of ellipsees, 
            Proc. of the 13th Internation Conference on Pattern Recognition, pp 253–257, Vienna, 1996
        Rewritten as an object, np.argmax(np.abs(E)) modified to np.argmax(E).
        """
        
        xnew=x-self.xmean
        ynew=y-self.ymean
        
        xnew = xnew[:,np.newaxis]
        ynew = ynew[:,np.newaxis]
        
        D = np.hstack((xnew*xnew, xnew*ynew, ynew*ynew, xnew, ynew, np.ones_like(xnew)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        #n = np.argmax(np.abs(E))
        n = np.argmax(E)
        
        self._parameters = V[:,n]
        a,b=self._calculate_axes_length_linalg()
        theta=self._calculate_angle_linalg()
        
        self._center=self._calculate_center_linalg()
        if a < b:
            self._axes_length=np.asarray([a,b])
            self._angle=np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=np.arcsin(np.sin(theta-np.pi/2))
        
    def _fit_ellipse_skimage(self,x,y):
        
        coordinates=np.vstack((x.ravel(),y.ravel())).transpose()
        try:
            ellipse=EllipseModel()
            ellipse.estimate(coordinates)
            xc, yc, a, b, theta = ellipse.params
        except:
            xc, yc, a, b, theta= [np.nan]*5
            
        self._center=np.asarray([xc,yc])
        if a < b:
            self._axes_length=np.asarray([a,b])
            self._angle=np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=np.arcsin(np.sin(theta-np.pi/2))

    def _calculate_angle_linalg(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        if b == 0:
            return 0.
        else:
            return np.arctan(2*b/(a-c))/2.
    
    def _calculate_axes_length_linalg(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1=(b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    def _calculate_center_linalg(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num+self.xmean
        y0=(a*f-b*d)/num+self.ymean
        return np.array([x0,y0])
        
    @property
    def angle(self):
        return self._angle
    
    @property
    def axes_length(self):
        """
        Returns the minor and major axes lengths of the ellipse, respectively.
        """
        return self._axes_length
    
    @property
    def center(self):
        """
        Returns the center of the ellipse.
        """
        return self._center
        
    @property
    def elongation(self):
        size=self.size
        return (size[0]-size[1])/(size[0]+size[1])
    
    @property
    def size(self):
        """
        Returns the size of the ellipse in the x and y direction along its center.
        Not part of the description in the source.
        a contains the coefficients like this:
        a[0]*x**2 + a[1]*x*y + a[2]*y**2 + a[3]*x + a[4]*y + a[5] = 0
        The sizes are calculated from the solution of the 2nd order equation:
        ysize=np.abs(y1-y2) where y1,y2 = y1,2(x=x0)
        xsize=np.abs(x1-x2) where x1,x2 = x1,2(y=y0)
        """

        alfa=self.angle
        a,b=self.axes_length
            
        a0=np.cos(alfa)**2/a**2 + np.sin(alfa)**2/b**2 #*x**2
        a1=2*np.cos(alfa)*np.sin(alfa)*(1/a**2-1/b**2) #*xy
        a2=np.sin(alfa)**2/a**2 + np.cos(alfa)**2/b**2 #*y**2
        
        xsize=2/np.sqrt(a0)
        ysize=2/np.sqrt(a2)
        
        # Old calculation
        # a=self._parameters
        # x0,y0=self.center
        
        # ax=a[0]
        # ay=a[2]
        # bx=a[1]*y0+a[3]
        # by=a[1]*x0+a[4]
        # cx=a[2]*y0**2+a[4]*y0+a[5]
        # cy=a[0]*x0**2+a[3]*x0+a[5]
        # xsize=np.sqrt(bx**2-4*ax*cx)/np.abs(ax)
        # ysize=np.sqrt(by**2-4*ay*cy)/np.abs(ay)
        
        if np.imag(xsize) != 0 or np.imag(xsize) !=0:
            print('size is complex')
            raise ValueError('')
        return np.array([xsize,ysize])

class FitGaussian:
    def __init__(self,
                 x=None,
                 y=None,
                 data=None):
        
        
        self._fwhm_to_sigma=(2*np.sqrt(2*np.log(2)))
        self.x=x
        self.y=y        
        self.data=data
        self.fit_gaussian(x,y,data)
        
        
    def fit_gaussian(self,x,y,data):
        xdata=np.vstack((x.ravel(),y.ravel()))        
        initial_guess=[data.max(),                                #Amplitude
                       np.sum(x*data)/np.sum(data),               #x0
                       np.sum(y*data)/np.sum(data),               #y0
                       (x.max()-x.min())/2/self._fwhm_to_sigma,   #Sigma_x
                       (y.max()-y.min())/2/self._fwhm_to_sigma,   #Sigma_y
                       0.,                                        #Angle
                       np.mean(data)                              #Offset
                       ]
        
        try:
            popt, pcov = scipy.optimize.curve_fit(gaussian2D_fit_function, 
                                                  xdata, 
                                                  data, 
                                                  p0=initial_guess)
            popt[5]=np.arcsin(np.sin(popt[5]))
            self.popt=popt
        except:
            self.popt=np.zeros(7)
            self.popt[:]=np.nan
            print('Gaussian fitting failed.')
        

        theta=self.popt[5]
        a,b=np.abs(np.asarray([self.popt[3],
                               self.popt[4]])*self._fwhm_to_sigma)
        
        if a < b:
            self._axes_length=np.asarray([a,b])
            self._angle=np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=np.arcsin(np.sin(theta-np.pi/2))
        
        self._center=np.asarray([self.popt[1],
                                 self.popt[2]])
        
    def set_invalid(self):
        self.popt=np.zeros(7)
        self.popt[:]=np.nan
    
    @property
    def angle(self):
        return self._angle
    
    @property
    def axes_length(self):
        return self._axes_length
    
    @property
    def center(self):
        return self._center
    
    @property
    def center_of_gravity(self):
        return np.asarray([np.sum(self.x*self.data)/np.sum(self.data),
                           np.sum(self.y*self.data)/np.sum(self.data)])
    
    @property
    def elongation(self):
        size=self.size
        return (size[0]-size[1])/(size[0]+size[1])
    
    @property
    def half_level(self):
        #WARNING: NOT PRESENT IN ELLIPSE
        return (self.popt[0]-self.popt[6])/2
    
    @property
    def size(self):
        
        alfa=self.angle
        a=self.axes_length[0]
        b=self.axes_length[1]
        
        a0=np.cos(alfa)**2/a**2 + np.sin(alfa)**2/b**2 #*x**2
        a1=2*np.cos(alfa)*np.sin(alfa)*(1/a**2-1/b**2) #*xy
        a2=np.sin(alfa)**2/a**2 + np.cos(alfa)**2/b**2 #*y**2

        xsize=2/np.sqrt(a0)
        ysize=2/np.sqrt(a2)

        if np.imag(xsize) != 0 or np.imag(xsize) !=0:
            raise ValueError('Size is complex')
            
        return np.array([xsize,ysize])
    
def gaussian2D_fit_function(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):

    x,y=coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = (np.sin(2*theta))/(4*sigma_x**2) - (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (+ a*(x-xo)**2 
                                      + 2*b*(x-xo)*(y-yo) 
                                      + c*(y-yo)**2)
                                  )
    return g.ravel()