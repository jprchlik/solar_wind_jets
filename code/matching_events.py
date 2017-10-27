import pandas as pd


#dictionary for storing the Pandas Data frame
plsm = {}



#set use to use all spacecraft
craft = ['wind','ace','dscovr','soho']

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
tr_events = tr_events[~tr_events.duplicated(['str_hour'],keep = 'first')]

#get strings for times around each event#
window = {}
window['dscovr'] = pd.to_timedelta('60 minutes')
window['ace'] = pd.to_timedelta('60 minutes')
window['soho'] = pd.to_timedelta('60 minutes')

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

        print(k)
        if p_mat.size > 0:
           #get the index of the maximum value
           i_max = p_mat[p_var].idxmax() 
           print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max={1:4.3f}'.format((i-i_max).total_seconds()/60.,p_mat.loc[i_max][p_var],i_max))
        else:
           print('No Observations')
    print('########################################')
        
    
