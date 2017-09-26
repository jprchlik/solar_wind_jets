import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 24

import pandas as pd
from fancy_plot import fancy_plot
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

d_fmt = '../{0}/data/y2016_formatted.pic'

sc = ['ACE','Wind','SOHO']
colors = ['red','black','teal','blue']
marker = ['o','D','s','^']

cdict = {}
bdict = {}


fig, ax = plt.subplots()


for j,i in enumerate(sc):

    df = pd.read_pickle(d_fmt.format(i.lower()))
    bdict[i] = df


    sig_vals = []
    sig_cnts = []
    for x in df.columns:
        if 'predict' in x:
            events, = np.where(df[x] > 0.990)
            sig_cnts.append(events.size)
            sig_vals.append(float(x.split('_')[-1])/100.)


    cdict[i] = [sig_cnts,sig_vals]

    ax.scatter(sig_vals,sig_cnts,marker=marker[j],color=colors[j],label=i)


ax.set_yscale('log')
ax.set_ylim([.5,2E4])
ax.set_xlim([2.5,10.0])
ax.set_ylabel('\# Events 2016')
ax.set_xlabel('N $\sigma$ Training')
ax.legend(loc='upper right',scatterpoints=1,frameon=False,handletextpad=-0.1)
fancy_plot(ax)



fig.savefig('../plots/all_num_events_sig_cut.png',bbox_pad=.1,bbox_inches='tight')
fig.savefig('../plots/all_num_events_sig_cut.eps',bbox_pad=.1,bbox_inches='tight')
