import pandas as pd 
import numpy as np
from fancy_plot import fancy_plot
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime,timedelta



shock_times = pd.read_table('shock_times.txt')
shock_times['start_time_dt'] = pd.to_datetime(shock_times.start_time)


#find all soho files in data directory
f_soho = glob('../soho/data/*txt')

#read in all soho files in data directory
df_file = (pd.read_table(f,skiprows=28,engine='python',delim_whitespace=True) for f in f_soho)

#create one large array with all soho information in range
soho_df = pd.concat(df_file,ignore_index=True)

#convert columns to datetime column
soho_df['time_dt'] = pd.to_datetime('20'+soho_df['YY'].astype('str')+':'+soho_df['DOY:HH:MM:SS'],format='%Y:%j:%H:%M:%S')

#set index to be time
soho_df.set_index(soho_df['time_dt'],inplace=True)


#Create variable where identifies shock
soho_df['shock'] = 0

#locate shocks and update parameter to 1
for i in shock_times.start_time_dt:
    #less than 120s seconds away from shock claim as part of shock (output in nano seconds by default)
    shock, = np.where(np.abs(((soho_df.time_dt-i).values/1.e9).astype('float')) < 120.)
    print soho_df['time_dt'][shock]
    soho_df['shock'][shock] = 1




#calculate difference in parameters
soho_df['del_time'] = soho_df['time_dt'].diff(-1).values.astype('double')/1.e9
soho_df['del_speed'] = soho_df['SPEED'].diff(-1)/soho_df.del_time
soho_df['del_Np'] = soho_df['Np'].diff(-1)/soho_df.del_time
soho_df['del_Vth'] = soho_df['Vth'].diff(-1)/soho_df.del_time


#mask for training set
train = ((soho_df.time_dt >= datetime(2016,06,5,0)) & (soho_df.time_dt <= datetime(2016,12,31,0)) & (soho_df.del_time < 60.))

#training set
soho_df_train = soho_df[train]#plot range 




fig,ax = plt.subplots(ncols=3)

#plot solar wind speed
ax[0].scatter(soho_df_train.del_speed,soho_df_train.shock,color='black')
ax[0].set_xlabel('$|\mathrm{V}|$ [km/s]')
fancy_plot(ax[0])

#plot solar wind density
ax[1].scatter(soho_df_train.del_Np,soho_df_train.shock,color='black')
ax[1].set_xlabel('n$_\mathrm{p}$ [cm$^{-3}$]')
ax[1].set_xscale('log')
fancy_plot(ax[1])

#Thermal Speed
ax[2].scatter(soho_df_train.Vth,soho_df_train.shock,color='black')
ax[2].set_xlabel('w$_\mathrm{p}$ [km/s]')
fancy_plot(ax[2])

plt.show()

