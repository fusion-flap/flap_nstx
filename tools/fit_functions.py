# -*- coding: utf-8 -*-

import numpy as np

def mtanh_func(x, a, b, c, h, xo):
    xp=2*(x - xo)/c
    return (h+b)/2 + (h-b)/2*((1 - a*xp)*np.exp(-xp) - np.exp(xp))/(np.exp(-xp) + np.exp(xp))

def mtanh_p_func(x, a, b, c, h, xo):
    xp=2*(x - xo)/c
    return ((h-b)*((2*a*xp-a-4)*np.exp(2*xp)-a))/(2*(np.exp(2*xp)+1)**2)

def mtanh_pp_func(x, a, b, c, h, xo):
    xp=2*(x - xo)/c
    return -(2*(h-b)*np.exp(2*xp)*((a*xp-a-2)*np.exp(2*xp)-a*xp-a+2))/(np.exp(2*xp)+1)**3

def mtanh_ppp_func(x, a, b, c, h, xo):
    xp=2*(x - xo)/c
    return (2*(h-b)*np.exp(2*xp)*((2*a*xp-3*a-4)*np.exp(4*xp)+(16-8*a*xp)*np.exp(2*xp)+2*a*xp+3*a-4))/(np.exp(2*xp)+1)**4

# a=0.08192631
# b=1.5534404e-13
# c=0.18028801
# h=0.42368126
# xo=0.93477042

# sol=root_scalar(mtanh_pp_func,args=(a,b,c,h,xo), method='brentq', x0=0.9,bracket=[0,1], fprime=mtanh_ppp_func)
# print(mtanh_p_func(sol.root, a, b, c, h, xo) )
# print(sol.root, sol.iterations, sol.function_calls)