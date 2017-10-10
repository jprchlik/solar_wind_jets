import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 24

import pandas as pd 
import numpy as np
from fancy_plot import fancy_plot
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import statsmodels.api as sm

##SET UP BOOLEANS
#Use my training days
mytrain = True 
#Use full DSCOVR mission files
full_DSCOVR = False
#create bokeh
create_bokeh = True 



#Use shock spotter shock times
shock_times = pd.read_table('shock_spotter_events.txt',delim_whitespace=True)
shock_times = shock_times[shock_times.P > .5]
shock_times['start_time_dt'] = pd.to_datetime(shock_times.YY.astype('str')+'/'+shock_times.MM.astype('str')+'/'+shock_times.DD.astype('str')+'T'+shock_times.HHMM.astype('str'),format='%Y/%b/%dT%H%M')
    


#format input dataframe
def format_df(inpt_df,span='3600s'):
    #Do parameter calculation for the 2016 year (training year)
    inpt_df['del_time'] = np.abs(inpt_df['time_dt'].diff(-1).values.astype('double')/1.e9)

    #calculate difference in parameters
    inpt_df['ldel_speed'] = np.abs(inpt_df['SPEED']*inpt_df['SPEED'].diff(-1)/inpt_df.del_time)
    inpt_df['ldel_Np'] = np.abs(inpt_df['Np'].diff(-1)/inpt_df.del_time)
    inpt_df['ldel_Vth'] = np.abs(inpt_df['Vth'].diff(-1)/inpt_df.del_time)

    #Find difference on otherside
    inpt_df['del_speed'] = np.abs(inpt_df['SPEED']*inpt_df['SPEED'].diff(1)/inpt_df.del_time)
    inpt_df['del_Np'] = np.abs(inpt_df['Np'].diff(1)/inpt_df.del_time)
    inpt_df['del_Vth'] = np.abs(inpt_df['Vth'].diff(1)/inpt_df.del_time)
    
    #calculate variance normalized parameters
    inpt_df['std_speed'] = inpt_df.SPEED*inpt_df.SPEED.rolling(span,min_periods=3).std()/inpt_df.del_time#/float(span[:-1])
    inpt_df['std_Np'] = inpt_df.Np.rolling(span,min_periods=3).std()/inpt_df.del_time#/float(span[:-1])
    inpt_df['std_Vth'] = inpt_df.Vth.rolling(span,min_periods=3).std()/inpt_df.del_time#/float(span[:-1])
    
    #Significance of the variation in the wind parameters
    inpt_df['sig_speed'] = inpt_df.del_speed/inpt_df.std_speed
    inpt_df['sig_Np'] = inpt_df.del_Np/inpt_df.std_Np
    inpt_df['sig_Vth'] = inpt_df.del_Vth/inpt_df.std_Vth
    
    #fill pesky nans and infs with 0 values
    inpt_df['sig_speed'].replace(np.inf,np.nan,inplace=True)
    inpt_df['sig_Np'].replace(np.inf,np.nan,inplace=True)
    inpt_df['sig_Vth'].replace(np.inf,np.nan,inplace=True)
    inpt_df['sig_speed'].fillna(value=0.0,inplace=True)
    inpt_df['sig_Np'].fillna(value=0.0,inplace=True)
    inpt_df['sig_Vth'].fillna(value=0.0,inplace=True)
    
    #create an array of constants that hold a place for the intercept 
    inpt_df['intercept'] = 1 
    return inpt_df




#read in full mission long DSCOVR information
if full_DSCOVR:
    #final all DSCOVR files in 30second cadence directory
    f_full = glob('../DSCOVR/data/30sec_cad/formatted_txt/*txt')
    #read in all DSCOVR files in 30sec_cad directory
    df_full = (pd.read_table(f,engine='python',delim_whitespace=True) for f in f_full)
    #create one large array with all DSCOVR information
    full_df = pd.concat(df_full,ignore_index=True)
    #only keep with values in the Time frame
    full_df = full_df[full_df['DOY:HH:MM:SS'] >  0]
    
    
    
    
    
    #convert columns to datetime column
    full_df['time_dt'] = pd.to_datetime(full_df['YY'].astype('int').map("{:02}".format)+':'+full_df['DOY:HH:MM:SS'],format='%y:%j:%H:%M:%S',errors='coerce')
    #full_df['time_str'] = full_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
    #set index to be time
    full_df.set_index(full_df['time_dt'],inplace=True)


#find all DSCOVR files in data directory
f_dscovr = '../dscovr/data/dscovr_h1_fc_2016.txt'

#read in all DSCOVR files in data directory
#df_file = ( for f in f_DSCOVR)
dscovr_df = pd.read_table(f_dscovr,engine='python',delim_whitespace=True)
#convert columns to datetime column
dscovr_df['time_dt'] = pd.to_datetime(dscovr_df['YEAR'].map('{:02}'.format)+':'+dscovr_df['MO'].map('{:02}'.format)+dscovr_df['DD'].map('{:02}'.format)+dscovr_df['HR'].map('{:02}'.format)+dscovr_df['MN'].map('{:02}'.format)+dscovr_df['SC'].map('{:02}'.format),format='%Y:%m%d%H%M%S')



dscovr_df['time_str'] = dscovr_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
#set index to be time
dscovr_df.set_index(dscovr_df['time_dt'],inplace=True)

#smooth to 1 minutes to removed small scale variation
dscovr_df['SPEED'] = np.sqrt(dscovr_df.Vx**2+dscovr_df.Vy**2+dscovr_df.Vz**2)


dscovr_df['shock'] = 0
#locate shocks and update parameter to 1
for i in shock_times.start_time_dt:
    #less than 120s seconds away from shock claim as part of shock (output in nano seconds by default)
    shock, = np.where(np.abs(((dscovr_df.time_dt-i).values/1.e9).astype('float')) < 10.)
    dscovr_df['shock'][shock] = 1



#sigma just range (range to scan sigmas in training set)
sig_jump = np.linspace(3,9,100)
sig_cnts = np.zeros(sig_jump.size)

#columns to use for training 
use_cols = ['sig_speed','sig_Np','sig_Vth','intercept']

span = '3600s'
dscovr_df = format_df(dscovr_df,span=span)

#locate shocks and update parameter to 1
for j,i in enumerate(sig_jump):
    #shock name with sigma values
    var = 'shock_{0:3.2f}'.format(i).replace('.','')

    #Create variable where identifies shock
    dscovr_df[var] = 0
    #Train that all events with a jump greater than n sigma per 30s are initially marked as shocks
    dscovr_df[var][((dscovr_df.sig_speed > i) | (dscovr_df.sig_Np > i) | (dscovr_df.sig_Vth > i))] = 1
    
    #build rough preliminary shock model based on observations
    try:
        logit = sm.Logit(dscovr_df[var],dscovr_df[use_cols])
        sh_rs = logit.fit()                                         #divide to get into usits of seconds
    except:
        sig_cnts[j] = np.nan
        continue
    #get predictions for full set
    dscovr_df['predict_'+var] = sh_rs.predict(dscovr_df[use_cols])
    events, = np.where(dscovr_df['predict_'+var] > 0.990)
    if events.size > 0: sig_cnts[j] = events.size
   


#save output
dscovr_df.to_pickle('../dscovr/data/y2016_formatted.pic')



#Do parameter calculation for all previous years 
#calculate difference in parameters
if full_DSCOVR:
    full_df = format_df(full_df,span=span)


#get predictions for the Mission long CELIAS mission
if full_DSCOVR: 
    full_df['predict'] = sh_rs.predict(full_df[use_cols])
    best_df = full_df[full_df.predict >= 0.90]
    best_df.to_csv('../DSCOVR/archive_shocks.csv',sep=';')


#create figure object
fig,ax = plt.subplots(figsize=(12,7))

#plot solar wind speed
ax.scatter(sig_jump,sig_cnts,color='black')
ax.set_xlabel(r'N sigma training')
ax.set_ylabel(r'\# of Events (2016)')
ax.set_yscale('log')
ax.set_ylim([.5,2E6])
fancy_plot(ax)

fig.savefig('../plots/dscovr_num_events_sig_cut.png',bbox_inches='tight',bbox_pad=0.1)
fig.savefig('../plots/dscovr_num_events_sig_cut.eps',bbox_inches='tight',bbox_pad=0.1)



###########
#BOKEH PLOT
#For the training set
###########
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show,save
from bokeh.layouts import column,gridplot


##########################################
#Create parameters for comparing data sets
##########################################
if create_bokeh:
    source = ColumnDataSource(data=dscovr_df[((dscovr_df.predict_shock_718 > .1) | (dscovr_df.shock == 1))])
    tools = "pan,wheel_zoom,box_select,reset,hover,save,box_zoom"
    
    tool_tips = [("Date","@time_str"),
                 ("Del. Np","@sig_Np"),
                 ("Del. Speed","@sig_speed"),
                 ("Del. Vth","@sig_Vth"),
                 ("Predict","@predict_shock_718"),
                 ]
    
    
    p1 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p1.scatter('sig_Np','sig_Vth',color='black',source=source)
    p1.select_one(HoverTool).tooltips = tool_tips
    p1.xaxis.axis_label = 'Delta Np/Np'
    p1.yaxis.axis_label = 'Delta Vth/Vth'
                                       
    p2 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p2.scatter('sig_Vth','sig_speed',color='black',source=source)
    p2.select_one(HoverTool).tooltips = tool_tips
    p2.xaxis.axis_label = 'Delta Vt/Vt'
    p2.yaxis.axis_label = 'Delta |V|/V'
                                       
    p3 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p3.scatter('sig_speed','sig_Np',color='black',source=source)
    p3.select_one(HoverTool).tooltips = tool_tips
    p3.xaxis.axis_label = 'Delta |V|/|V|'
    p3.yaxis.axis_label = 'Delta Np/Np'
                                       
    p4 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p4.scatter('shock','predict_shock_718',color='black',source=source)
    p4.select_one(HoverTool).tooltips = tool_tips
    p4.xaxis.axis_label = 'DSCOVR DB SHOCK'
    p4.yaxis.axis_label = 'My Shock'
                                       
    p5 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p5.scatter('SPEED','Np',color='black',source=source)
    p5.select_one(HoverTool).tooltips = tool_tips
    p5.xaxis.axis_label = '|V| [km/s]'
    p5.yaxis.axis_label = 'Np [cm^-3]'

    p6 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p6.scatter('SPEED','Vth',color='black',source=source)
    p6.select_one(HoverTool).tooltips = tool_tips
    p6.xaxis.axis_label = '|V| [km/s]'
    p6.yaxis.axis_label = 'Vth [km/s]'
                                       
    p7 = figure(title='DSCOVR CELIAS SHOCKS',tools=tools)
    p7.scatter('Np','Vth',color='black',source=source)
    p7.select_one(HoverTool).tooltips = tool_tips
    p7.xaxis.axis_label = 'Np [cm^-3]'
    p7.yaxis.axis_label = 'Vth [km/s]'



    save(gridplot([p1,p2],[p3,p4],[p5,p6],[p7,]),filename='../plots/bokeh_training_plot_dscovr.html')
    
