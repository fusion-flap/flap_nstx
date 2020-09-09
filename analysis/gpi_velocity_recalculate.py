#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:46:55 2020

@author: mlampert
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:06:17 2020

@author: mlampert
"""


import pandas
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

import numpy as np

import os
flap_nstx.register()
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#elms=np.asarray([[139962, 0.3945],
#      [139962, 0.4062],
#      [139965, 0.3915],
#      [139969, 0.4095],
#      [140616, 0.4335],
#      [140620, 0.5574],
#      [141303, 0.527],
#      [141303, 0.5432],
#      [141307, 0.5105],
#      [141307, 0.5133],
#      [141307, 0.503],
#      [141307, 0.499],
#      [141310, 0.498],
#      [141310, 0.5085],
#      [141311, 0.495],
#      [141313, 0.5157],
#      [141314, 0.493],
#      [141314, 0.5165],
#      [141326, 0.5607],
#      [141448, 0.483],
#      [141747, 0.257],
#      [141919, 0.2427],
#      [141920, 0.2777],
#      [141920, 0.2787],
#      [141922, 0.261],
#      [142006, 0.2467],
#      [142231, 0.430],
#      [142231, 0.446],
#      [142232, 0.394],
#      [142232, 0.4123]])

elms=np.asarray([[139878,329.3],
                 [139896,343.0],
                 [139901,307.5],
                 [139906,307.1],
                 [139907,334.7],
                 [139907,340.7],
                 [140620,556.5],
                 [141301,477.5]])


elm=0.
failed_elms=[]
number_of_failed_elms=0
for i_elm in range(elms.shape[0]):
    shot=int(elms[i_elm,0])
    elm_time=elms[i_elm,1]/1000
    flap.delete_data_object('*')

    print('Calculating '+str(shot)+ ' at '+str(elm_time))
    start_time=time.time()
    try:
        calculate_nstx_gpi_avg_frame_velocity(exp_id=shot, 
                                              time_range=[elm_time-2e-3,elm_time+2e-3], 
                                              plot=False,
                                              subtraction_order_for_velocity=1, 
                                              pdf=True, 
                                              nlevel=51, 
                                              nocalc=False, 
                                              filter_level=3, 
                                              normalize_for_size=True,
                                              normalize_for_velocity=False,
                                              threshold_coeff=1.,
                                              normalize_f_high=1e3, 
                                              normalize='roundtrip', 
                                              velocity_base='cog', 
                                              return_results=False, 
                                              plot_gas=True)
    except:
        print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
        failed_elms.append({'Shot':shot,'Time':elm_time})
        number_of_failed_elms+=1
    one_time=time.time()-start_time
    rem_time=one_time*(30-i_elm)
    print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')
    print(failed_elms,number_of_failed_elms)
    
    
#FAILED SHOTS: [{'Shot': 138748, 'Time': <module 'time' (built-in)>}, {'Shot': 139897, 'Time': <module 'time' (built-in)>}, {'Shot': 139913, 'Time': <module 'time' (built-in)>}, {'Shot': 139913, 'Time': <module 'time' (built-in)>}, {'Shot': 141920, 'Time': <module 'time' (built-in)>}, {'Shot': 142231, 'Time': <module 'time' (built-in)>}] 6    