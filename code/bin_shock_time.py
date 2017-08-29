import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 24

import pandas as pd
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot


#read in shock file
shock_df = pd.read_csv('../soho/archive_shocks.csv',sep=';')
shock_df['time_dt'] = pd.to_datetime(shock_df['time_dt'])
shock_df.set_index(shock_df['time_dt'],inplace=True)

#number of events
shock_df['counts'] = 1

#Resample at one month using the mean
df_bin = shock_df.resample('12M').sum()



fig, ax = plt.subplots(figsize=(8,8))

ax.scatter(df_bin.index,df_bin.counts)
ax.set_ylabel('Number of Shock Events')
ax.set_xlabel('Year')
fancy_plot(ax)

plt.show()
