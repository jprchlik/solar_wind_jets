import pandas as pd
import matplotlib.pyplot as plt


#dictionary for storing the Pandas Data frame
plsm = {}


#use space craft position to get range of time values
use_craft = False

#use best chisq to find the best time in between spacecraft
use_chisq = True


#set use to use all spacecraft
craft = ['wind','dscovr','ace','soho']

#space craft to use n sigma events to find other events
#change craft order to change trainer 
trainer = craft[0]


#sigma level to use from wind training
sig_l = 5.0
#use sig_l to create shock prediction variable in dataframes
p_var = 'predict_shock_{0:3.2f}'.format(sig_l).replace('.','')
#fractional p value to call an "event"
p_val = 0.999

#read in all spacraft events
for k in craft: plsm[k] = pd.read_pickle('../{0}/data/y2016_power_formatted.pic'.format(k))



#use the trainer space craft to find events
tr_events = plsm[trainer][plsm[trainer][p_var] > p_val]

#Group events by 1 per hour
tr_events['str_hour'] = tr_events.time_dt.dt.strftime('%Y%m%d%H')
#attempting to pick one event at a time J. Prchlik (2017/10/30) (Did not work as 1pm of the same day)
#tr_events = tr_events[~tr_events.duplicated(['str_hour'],keep = 'first')]
#loop over events and get best probability for event within 1 hour
tr_events['group'] = range(tr_events.index.size)
for i in tr_events.index:
    tr_events['t_off'] = abs((i-tr_events.index).total_seconds())
    match = tr_events[tr_events['t_off'] < 3600.] #anytime where the offest is less than 1 hour 
    tr_events.loc[match.index,'group'] = match.group.values[0]

#use groups to cut out duplicates
tr_events = tr_events[~tr_events.duplicated(['group'],keep = 'first')]

#get strings for times around each event#
window = {}
window['dscovr'] = pd.to_timedelta('90 minutes')
window['ace'] = pd.to_timedelta('90 minutes')
window['soho'] = pd.to_timedelta('90 minutes')
window['wind'] = pd.to_timedelta('90 minutes')

#space craft to match with trainer soho
craft.remove(trainer)

#use wind speed to esimate possible time delay
min_vel = 300. #km/s
max_vel = 2000. #km/s
Re = 6378. #Earth radius in km

#convert ace coordinates into Re
plsm['ace']['X'] = plsm['ace'].X/Re

#get event slices 
for i in tr_events.index:
    print('########################################')
    print('NEW EVENT')
    print(trainer)
    print('{0:%Y/%m/%d %H:%M:%S}, p={1:4.3f}'.format(i,tr_events.loc[i,p_var]))
    #get time slice around event

    #loop over all other space craft
    for k in craft:


        #initial time slice to get the approximate spacecraft differences
        init_slice = [i-window[k],i+window[k]] 
        p_mat = plsm[k].loc[init_slice[0]:init_slice[1]] 

        #print current spacecraft
        print(k)
        #use spacecraft position to get solar wind times for matching
        if use_craft:

            
  
            #find the median distance
            #med_dis = np.median(np.sqrt((p_mat.X-i.X)**2.+(p_mat.Y-i.Y)**2.+(p_mat.Z-i.Z)**2.))
            #Just use X distance since most of wind is in that direction
            #towards the sun is positive X (GSE)
            #use negative sign to change to increasing away from the sun
            med_dis = (p_mat.X-tr_events.loc[i,'X']).median()

            #convert the median distance into a time delay (distances in Re)
            if med_dis > 0:
                max_del = med_dis*Re/min_vel
                min_del = med_dis*Re/max_vel
            else:
                min_del = med_dis*Re/min_vel
                max_del = med_dis*Re/max_vel
            

            #convert to pandas time delta window
            cal_wd = pd.to_timedelta([min_del,max_del],unit='s')
     
            #get a time slice base on space craft differences
            time_slice = [i+cal_wd[0],i+cal_wd[1]] 
            if p_mat.size > 0:
               #get the index of the maximum value
               i_max = p_mat[p_var].idxmax() 
               print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max={1:4.3f}'.format((i_max-i).total_seconds()/60.,p_mat.loc[i_max][p_var],i_max))
            else:
               print('No Plasma Observations')
  
        #use chisq minimum of top events in 2 hour window
        elif use_chisq:
            #magnetic field model fitting
            p_mag = p_mat.sort_values(p_var.replace('predict','predict_sigma'),ascending=False)[0:10]
            #sort the cut window and get the top 10 events
            p_mat = p_mat.sort_values(p_var,ascending=False)[0:10]


            if p_mat.size > 0:

                #create figure to test fit
                fig, ax = plt.subplots()
                ax.set_title(k)

                #set up chisquared array in pandas object
                p_mat['chisq'] = -99999.9
                for time in p_mat.index:
                    #get a region around one of the best fit times
                    com_slice = [time-window[k],time+window[k]]
                    c_mat = plsm[k].loc[com_slice[0]:com_slice[1]]

                    #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
                    c_mat.index = c_mat.index+(i-time)

                    #get trainint spacecraft time range
                    t_mat = plsm[trainer].loc[c_mat.index.min():c_mat.index.max()]

                    #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid and interpolate
                    c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')

                    #compute the chisq value in SPEED from the top ten probablilty array
                    p_mat.loc[time,'chisq'] = (sum((c_mat.SPEED-t_mat.SPEED)**2.))**.5

                    #create figure to check matching
                    ax.scatter(c_mat.index,c_mat.SPEED,label=time.to_datetime().strftime('%Y/%m/%dT%H:%M:%S')+' chisq = {0:4.0f}'.format(p_mat.loc[time,'chisq']))
                    ax.plot(t_mat.index,t_mat.SPEED,label='',color='black')

                ax.legend(loc='upper left',frameon=False,scatterpoints=1)
                ax.set_xlim([t_mat.index.min(),t_mat.index.max()])
                ax.set_ylabel('Speed [km/s]')
                ax.set_xlabel('Time [UTC]')
                ax.set_ylim([300,1000.])
                plt.show()
     
                #get the index of minimum chisq value
                i_min = p_mat['chisq'].idxmin()
                if k == 'soho': print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min))
                else: print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
            #else no plasma observations                                                                                                                                                      
            elif ((p_mat.size == 0) & (p_mag.size > 0.)):
               if k != 'soho':
                   #sort the cut window and get the top 10 events
                   p_mat = p_mag
       
                   if p_mat.size > 0:
                       #set up chisquared array in pandas object
                       p_mat['chisq'] = -99999.9
                       for time in p_mat.index:
                           #get a region around one of the best fit times
                           com_slice = [time-window[k],time+window[k]]
                           c_mat = plsm[k].loc[com_slice[0]:com_slice[1]]
       
                           #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
                           c_mat.index = c_mat.index+(i-time)
       
                           #get trainint spacecraft time range
                           t_mat = plsm[trainer].loc[c_mat.index.min():c_mat.index.max()]
       
                           #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid and interpolate
                           c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')
       
                           #compute the chisq value in SPEED from the top ten probablilty array
                           p_mat.loc[time,'chisq'] = (sum((c_mat.Bx-t_mat.Bx)**2.+(c_mat.By-t_mat.By)**2.+(c_mat.Bz-t_mat.Bz)**2.))**.5
            
                       #get the index of minimum chisq value
                       i_min = p_mat['chisq'].idxmin()
                       print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
       
            else:
               print('No Plasma or Mag. Observations')

        #print separater 
        print('########################################')
    
