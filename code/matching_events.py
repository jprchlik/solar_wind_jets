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

#get strings for times around each event#
window = {}
window['dscovr'] = pd.to_timedelta('40 minutes')
window['ace'] = pd.to_timedelta('60 minutes')
window['soho'] = pd.to_timedelta('60 minutes')

#space craft to match with trainer soho
craft.remove(trainer)

#get event slices 
for i in tr_events.index:
    print('###############################')
    print('NEW EVENT')
    print(trainer)
    print('{0:%Y/%m/%d %H:%M:%S}, p={1:4.3f}'.format(i,tr_events.loc[i,p_var]))
    #get time slice around event

    #loop over all other space craft
    for k in craft:
        time_slice = [i-window[k],i+window[k]] 
        p_mat = plsm[k].loc[time_slice[0]:time_slice[1]] 


        print(k)
        if p_mat.size > 0:
           #get the index of the maximum value
           i_max = p_mat[p_var].idxmax() 
           print('{0:5.2f} min., p_max={1:4.3f}'.format((i-i_max).total_seconds()/60.,p_mat.loc[i_max][p_var]))
        else:
           print('No Observations')
    print('###############################')
        
    
