import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from numpy import linspace,where
import pandas as pd

smooth = True

#set use to use all spacecraft
craft = ['DSCOVR','Wind','ACE','SOHO']
color = ['black', 'blue','red','teal']
symbl = ['o','D','s','<']

#sigma level to use from wind training
sig_l = 5.0
#use sig_l to create shock prediction variable in dataframes
p_var = 'predict_shock_{0:3.2f}'.format(sig_l).replace('.','')
#fractional p value to call an "event"
p_val = linspace(0.68,0.9999)

#plot for comparing number of "events" distributions
fig, ax =plt.subplots()

#read in all spacraft events
for j,k in enumerate(craft): 
    if smooth: plsm = pd.read_pickle('../{0}/data/y2016_power_smoothed_formatted.pic'.format(k.lower()))
    else: plsm = pd.read_pickle('../{0}/data/y2016_power_formatted.pic'.format(k.lower()))

    #Group events by 1 per hour
    plsm['str_hour'] = plsm.time_dt.dt.strftime('%Y%m%d%H')
    plsm = plsm[~plsm.duplicated(['str_hour'],keep = 'first')]

    counts = []
    #loop over all p-values an count the number of events
    for i in p_val:
        events, = where(plsm[p_var] >= i)
        counts.append(events.size)
    ax.scatter(p_val,counts,marker=symbl[j],color=color[j],label=k)

ax.set_yscale('log')
ax.set_ylim([0.5,2E2])
ax.set_xlabel('Event Probability')
ax.set_ylabel('Number of Events')

ax.legend(loc='upper right',frameon=False,scatterpoints=1,handletextpad=-0.112)
fancy_plot(ax)

if smooth:
   fig.savefig('../plots/p_dis_four_craft_smoothed.png',bbox_pad=.1,bbox_inches='tight')
   fig.savefig('../plots/p_dis_four_craft_smoothed.eps',bbox_pad=.1,bbox_inches='tight')
else:
   fig.savefig('../plots/p_dis_four_craft.png',bbox_pad=.1,bbox_inches='tight')
   fig.savefig('../plots/p_dis_four_craft.eps',bbox_pad=.1,bbox_inches='tight')