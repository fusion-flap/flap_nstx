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

class FitEllipse:

    def __init__(self,
                 x=None,                                                        #The x coordinates of the input data as a numpy array
                 y=None,                                                        #The y coordinates of the input data as a numpy array
                 method='linalg',                                                #linalg, skimage, leastsquare, or linalg_v0 (deprecated)
                 elongation_base='size',                                        #size or axes
                 test=False,
                 ):

        if method not in ['linalg',
                          'skimage',
                          'leastsquare',
                          'linalg_v0']:
            print(['linalg',
                   'skimage',
                   'leastsquare',
                   'linalg_v0'])
            raise ValueError('\n ^^^ The method needs to be either one of these ^^^!')

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
        self._xmean=np.mean(x)
        self._ymean=np.mean(y)
        self._elongation_base=elongation_base
        self._test=test

        try:
        #if True:

            if method=='linalg':
                self._fit_ellipse_linalg(x, y)
            elif method=='skimage':
                self._fit_ellipse_skimage(x, y)
            elif method=='leastsquare':
                self._fit_ellipse_leastsq(x, y)
            if method=='linalg_v0':
                self._fit_ellipse_linalg_v0(x, y)
        except Exception as e:
            print(e)
            self.set_invalid()


    def set_invalid(self):

        self._angle=np.nan
        self._axes_length=np.asarray([np.nan,np.nan])
        self._center=np.asarray([np.nan,np.nan])
        self._parameters=np.asarray([np.nan]*6)

        print('Ellipse fitting failed')

    def _fit_ellipse_linalg(self, x, y):
        """

        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.

        https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

        """

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        self._parameters=np.concatenate((ak, T @ ak)).ravel()

        self._center=self._calculate_center_linalg()
        self._axes_length=self._calculate_axes_length_linalg()
        self._angle=self._calculate_angle_linalg()

    def _calculate_angle_linalg(self):
        p=self._parameters
        a,b,c,d,f,g=p[0],p[1]/2,p[2],p[3]/2,p[4]/2,p[5]

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        try:
            self._width_gt_height
        except:
            self._calculate_axes_length_linalg()
        if not self._width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        # phi = phi % np.pi
        #Adjustment to be between [-pi/2,pi/2]
        if phi > np.pi/2:
            while phi > np.pi/2:
                phi -= np.pi
        else:
            while phi < -np.pi/2:
                phi += np.pi

        return phi


    def _calculate_axes_length_linalg(self):
        p=self._parameters
        a,b,c,d,f,g=p[0],p[1]/2,p[2],p[3]/2,p[4]/2,p[5]
        den = b**2 - a*c
        if den > 0:
            raise ValueError('Coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        self._width_gt_height = True
        if ap < bp:
            self._width_gt_height = False
            ap, bp = bp, ap
        return np.asarray([ap,bp])

    def _calculate_center_linalg(self):
        p=self._parameters
        a,b,c,d,f,g=p[0],p[1]/2,p[2],p[3]/2,p[4]/2,p[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('Coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')

        # The location of the ellipse centre.
        return np.array([(c*d - b*f) / den,
                         (a*f - b*d) / den])

    def _fit_ellipse_leastsq(self,x,y):
        """
        Source: https://stackoverflow.com/questions/67537630/ellipse-fitting-to-determine-rotation-python

        Realizes a simple least square fit. It needs to be tested, because there is no test for the
        structure/fitting being a hyperbole.

        Parameters
        ----------
        x : ndarray
            x coordinates of the structure to be fit
        y : ndarray
            y coordinates of the structure to be fit

        Returns
        -------
        None.

        """

        aat=np.zeros([5,5])

        aat[0,:]=[np.sum(x**4),         np.sum(2 * x**3 * y),    np.sum(x**2 * y**2),  np.sum(2 * x**3),     np.sum(2 * x**2 * y)]
        aat[1,:]=[np.sum(2 * x**3 * y), np.sum(4 * x**2 * y**2), np.sum(2 * x * y**3), np.sum(4 * x**2 * y), np.sum(4 * x * y**2)]
        aat[2,:]=[np.sum(x**2 * y**2),  np.sum(2 * x * y**3),    np.sum(y**4),         np.sum(2 * x * y**2), np.sum(2 * y**3)]
        aat[3,:]=[np.sum(2 * x**3),     np.sum(4 * x**2 * y),    np.sum(2 * x * y**2), np.sum(4 * x**2),     np.sum(4 * x * y)]
        aat[4,:]=[np.sum(2 * x**2 * y), np.sum(4 * x * y**2),    np.sum(2 * y**3),     np.sum(4 * x * y),    np.sum(4 * y**2)]

        coord_vec=np.asarray([np.sum(x**2), np.sum(2*x*y), np.sum(y**2), np.sum(2*x), np.sum(2*y)])

        self._parameters=np.matmul(np.linalg.inv(aat), coord_vec)

        self._axes_length=self._calculate_axes_length_leastsq()
        self._angle=self._calculate_angle_leastsq()
        self._center=self._calculate_center_leastsq()

        if self._test:
            print('parameters', self._parameters)
            print('axes',self._axes_length)
            print('angle',self._angle)
            print('center',self._center)

    def _calculate_angle_leastsq(self):
        p=self._parameters
        a,b,c,d,f = p[0],p[1],p[2],p[3],p[4]

        return np.pi/2 + 0.5*np.arctan2(2*b,a-c)

    def _calculate_center_leastsq(self):
        p=self._parameters
        a,b,c,d,f = p[0],p[1],p[2],p[3],p[4]

        x0=(c*d-b*f)/(b**2 - a*c)
        y0=(a*f-b*d)/(b**2 - a*c)
        return np.array([x0,y0])

    def _calculate_axes_length_leastsq(self):
        p=self._parameters
        a,b,c,d,f,g = p[0],p[1],p[2],p[3],p[4],-1


        nom=2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        denom_1=(b**2 - a*c) * (np.sqrt((a-c)**2 + 4 * b**2) - (a+c))
        A=np.sqrt(nom/denom_1)
        denom_2=(b**2 - a*c) * (-np.sqrt((a-c)**2 + 4 * b**2) - (a+c))
        B=np.sqrt(nom/denom_2)

        return np.asarray([A,B])

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
            self._angle=theta #np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=theta #np.arcsin(np.sin(theta-np.pi/2))

    def _fit_ellipse_linalg_v0(self,x,y):
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

        print("This version of the ellipse fitting is deprecated \
              because of the 90degree angle fitting issue. \
                  Please use method='linalg' instead of linalg_v0")


        xnew=x-self._xmean
        ynew=y-self._ymean

        xnew = xnew[:,np.newaxis]
        ynew = ynew[:,np.newaxis]

        D = np.hstack((xnew*xnew, xnew*ynew, ynew*ynew, xnew, ynew, np.ones_like(xnew)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        #n = np.argmax(E)

        self._parameters = V[:,n]


        a,b=self._calculate_axes_length_linalg_v0()
        theta=self._calculate_angle_linalg_v0()
        self._center=self._calculate_center_linalg_v0()
        # theta=self._angle

        if a < b:
            self._axes_length=np.asarray([a,b])
            self._angle=np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=np.arcsin(np.sin(theta))+np.pi/2

    def _calculate_angle_linalg_v0(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        if b == 0:
            return 0.
        else:
            return np.arctan(2*b/(a-c))/2.

    def _calculate_axes_length_linalg_v0(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1=(b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])

    def _calculate_center_linalg_v0(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num+self._xmean
        y0=(a*f-b*d)/num+self._ymean
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
    def axes(self):
        return self._axes_length

    @property
    def center(self):
        """
        Returns the center of the ellipse.
        """
        return self._center

    @property
    def elongation(self):
        if self._elongation_base=='size':
            size=self.size
            return (size[0]-size[1])/(size[0]+size[1])

        elif self._elongation_base=='axes':
            axes=self.axes
            return (axes[0]-axes[1])/(axes[0]+axes[1])

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

        if np.imag(xsize) != 0 or np.imag(ysize) !=0:
            print('size is complex')
            xsize=np.nan
            ysize=np.nan
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