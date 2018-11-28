import model_time_range as mtr
from scipy.io import readsav
from fancy_plot import fancy_plot
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd



#Just do 1 event for now 2018/04/25 J. Prchlik 
mike_event = ['../mikes_big_events/event_20170105.idl']

#mike keys in the order I use them (Wind,DSCOVR,ACE,SOHO)
skey = ['WI','DS','AC','SO']

#positions to get from orbital array
pkey = ['TIME','X','Y','Z']

#Velocities from solution to get from mike array
vkey = ['TIME','V','VX','VY','VZ','N','W']


#Wind to get DTW solution
window = pd.to_timedelta(1.*3600.,unit='s')



#Setup format for datetime string to pass to my_dtw later
dfmt = '{0:%Y/%m/%d %H:%M:%S}'

#twind = pd.to_datetime('2016/12/20 15:38:38')    
#twind = pd.to_datetime('2016/12/21 08:43:12')    
#twind = pd.to_datetime('2016/07/28 13:25:21')    
#twind = pd.to_datetime('2017/07/01 16:55:03')    
#twind = pd.to_datetime('2016/10/12 21:16:14')    
#twind = pd.to_datetime('2016/12/09 04:45:29')    
#twind = pd.to_datetime('2016/12/21 08:43:12')    

twinds = [
          pd.to_datetime('2016/10/12 21:16:14')
          ]   



for twind in twinds:

    start_t = dfmt.format(twind-2.0*window)
    #For the 2016/12/21 event
    end_t = dfmt.format(twind+2.5*window)
    #2016/12/09 04:45:29


    
    #my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B'],penalty=False)
    #my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B'],penalty=False,events=2)
    
    #'2016/10/12 21:16:14
    #my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B','THEMIS_C'],penalty=True,events=3)
    my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,penalty=True,mag_pen=.15,par=['SPEED'])
    
    #2016/12/21
    #my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B','THEMIS_C'],penalty=True,events=7)
    #my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B'],penalty=True,events=7)
    
    
    
    my_dtw.init_read()  
    
    #Do not use multi parm Single parameter solution works better
    my_dtw.dtw()
    my_dtw.fig.savefig('../plots/bou_{0:%Y%m%d_%H%M%S}.png'.format(pd.to_datetime(twind)),bbox_pad=.1,bbox_inches='tight')
    #####my_dtw.dtw_multi_parm(twind)
    #####my_dtw.pred_earth()
    ####
    ####vn = my_dtw.event_dict['event_1']['vn'].T #The normal vectory for my solution
    ####
    #####Get my tvals and switch to mike's order (Wind,DSCOVR,SOHO,ACE)
    ####tvals = my_dtw.event_dict['event_1']['tvals']
    ####
    ####
    #####Create plot comparing OMNI and plane solution
    ####mtr.omni_plot(my_dtw)


    #####show example of DTW in action
    #####mtr.plot_dtw_example(my_dtw,twind)
    ####mtr.plot_dtw_example(my_dtw,twind+pd.to_timedelta('14m'),pad=pd.to_timedelta('18m'),subsamp=15)
