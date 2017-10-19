import pandas as pd


#dictionary for storing the Pandas Data frame
plsm = {}



#set use to use all spacecraft
craft = ['wind','ace','dscovr','soho']

#space craft to use n sigma events to find other events
#change craft order to change trainer 
trainer = craft[0]

#sigma level to use from wind training
sig_l = 4.5
#use sig_l to create shock prediction variable in dataframes
p_var = 'predict_shock_{0:3.2f}'.format(sig_l).replace('.','')
#fractional p value to call an "event"
p_val = 0.68

#read in all spacraft events
for k in craft: plsm[k] = pd.read_pickle('../{0}/data/y2016_power_formatted.pic'.format(k))



#use the trainer space craft to find events
tr_events = plsm[trainer][plsm[trainer][p_var] > p_val]

#get strings for times around each event#
window = pd.to_timedelta('60 minutes')


#get event slices 
