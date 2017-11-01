import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from numpy import linspace,where
import pandas as pd

smooth = False

#set use to use all spacecraft
craft = ['DSCOVR','Wind','ACE','SOHO']
color = ['black', 'blue','red','teal']
symbl = ['o','D','s','<']

#sigma level to use from wind training
sig_l = 5.0
#use sig_l to create shock prediction variable in dataframes
p_var = 'predict_shock_{0:3.2f}'.format(sig_l).replace('.','')
#fractional p value to call an "event"
p_val = linspace(0.68,0.9999,50)

#plot for comparing number of "events" distributions
fig, ax =plt.subplots()

#read in all spacraft events
for j,k in enumerate(craft): 
    if smooth: plsm = pd.read_pickle('../{0}/data/y2016_power_smoothed_formatted.pic'.format(k.lower()))
    else: plsm = pd.read_pickle('../{0}/data/y2016_power_formatted.pic'.format(k.lower()))

    #Group events by 1 per hour
    #Added acceleration parameter to hopefully beat back 
#    plsm['str_hour'] = plsm.time_dt.dt.strftime('%Y%m%d%H')
#    plsm = plsm[~plsm.duplicated(['str_hour'],keep = 'first')]

    #resample to 30 minutes because that is all that is needed
    tplms = plsm.resample('30min',label='right').max()
    counts = []
    #loop over all p-values an count the number of events
    for i in p_val:
        #get events at p_value level
        tr_events = tplms[tplms[p_var] > i]
        tr_events['group'] = range(tr_events.index.size)

        #update to rolling hour matching for events
        for i in tr_events.index:
            tr_events['t_off'] = abs((i-tr_events.index).total_seconds())
            match = tr_events[tr_events['t_off'] < 3600.] #anytime where the offest is less than 1 hour 
            tr_events.loc[match.index,'group'] = match.group.values[0]

        #use groups to cut out duplicates
        tr_events = tr_events[~tr_events.duplicated(['group'],keep = 'first')]

        #Count number after regrouping
        counts.append(tr_events.index.size)
    #plot counds as a function of the p value
    ax.scatter(p_val,counts,marker=symbl[j],color=color[j],label=k)

ax.set_yscale('log')
ax.set_ylim([0.5,2E3])
ax.set_xlabel('Event Probability')
ax.set_ylabel('Number of Events')

ax.legend(loc='lower left',frameon=False,scatterpoints=1,handletextpad=-0.112)
fancy_plot(ax)

if smooth:
   fig.savefig('../plots/p_dis_four_craft_smoothed.png',bbox_pad=.1,bbox_inches='tight')
   fig.savefig('../plots/p_dis_four_craft_smoothed.eps',bbox_pad=.1,bbox_inches='tight')
else:
   fig.savefig('../plots/p_dis_four_craft.png',bbox_pad=.1,bbox_inches='tight')
   fig.savefig('../plots/p_dis_four_craft.eps',bbox_pad=.1,bbox_inches='tight')