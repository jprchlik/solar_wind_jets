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
#Use full soho mission files
full_soho = False
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

    #get the Energy Change per second
    inpt_df['power'] = inpt_df.del_Np*inpt_df.SPEED**2.+2.*inpt_df.del_speed*inpt_df.Np*inpt_df.SPEED
    inpt_df['Np_power'] = inpt_df.del_Np*inpt_df.SPEED**2.
    inpt_df['speed_power'] = 2.*inpt_df.del_speed*inpt_df.Np*inpt_df.SPEED
    inpt_df['Npth_power'] = inpt_df.del_Np*inpt_df.Vth**2.
    inpt_df['Vth_power'] = 2.*inpt_df.del_Vth*inpt_df.Np*inpt_df.Vth

    #absolute value of the power
    inpt_df['abs_power'] = np.abs(inpt_df.power)
    inpt_df['Np_abs_power'] = np.abs(inpt_df.Np_power)
    inpt_df['speed_abs_power'] =  np.abs(inpt_df.speed_power)
    inpt_df['Npth_abs_power'] = np.abs(inpt_df.Npth_power)
    inpt_df['Vth_abs_power'] =  np.abs(inpt_df.Vth_power)

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




#read in full mission long soho information
if full_soho:
    #final all soho files in 30second cadence directory
    f_full = glob('../soho/data/30sec_cad/formatted_txt/*txt')
    #read in all soho files in 30sec_cad directory
    df_full = (pd.read_table(f,engine='python',delim_whitespace=True) for f in f_full)
    #create one large array with all soho information
    full_df = pd.concat(df_full,ignore_index=True)
    #only keep with values in the Time frame
    full_df = full_df[full_df['DOY:HH:MM:SS'] >  0]
    
    
    
    
    
    #convert columns to datetime column
    full_df['time_dt'] = pd.to_datetime(full_df['YY'].astype('int').map("{:02}".format)+':'+full_df['DOY:HH:MM:SS'],format='%y:%j:%H:%M:%S',errors='coerce')
    #full_df['time_str'] = full_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
    #set index to be time
    full_df.set_index(full_df['time_dt'],inplace=True)


#set use to use all spacecraft
craft = ['wind','ace','dscovr','soho']

#space craft to use n sigma events to train power on other spacecraft
#change craft order to change trainer 
trainer = craft[0]


for k in craft:

    #for reading in each space craft separately 
    if k == 'soho':
        #find all soho files in data directory
        f_soho = glob('../soho/data/*txt')
        #read in all soho files in data directory
        df_file = (pd.read_table(f,skiprows=28,engine='python',delim_whitespace=True) for f in f_soho)
        
        #create one large array with all soho information in range
        plms_df = pd.concat(df_file,ignore_index=True)
        
        #convert columns to datetime column
        plms_df['time_dt'] = pd.to_datetime('20'+plms_df['YY'].astype('str')+':'+plms_df['DOY:HH:MM:SS'],format='%Y:%j:%H:%M:%S')
        plms_df['time_str'] = plms_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
    #read in ace data
    elif k == 'ace': 
        f_aces = '../ace/data/ACE_swepam_level2_data_64sec_2016.txt'
        
        aces_nm = ['year','day','hour','minute','num_imf_ave','per_interp','cpmv_flag',
                   'time_shift','phi_norm_x','phi_norm_y','phi_norm_z','mag_b','Bx','By',
                   'Bz','s_By','s_Bz','rms_time_shift','rms_phase_front','rms_mag_b',
                   'rms_field_vec','num_plasma_ave','f_speed','Vx','Vy','Vz','Np','Tth',
                   'X','Y','Z','Xt','Yt','Zt','rms','dbot1','dbot2']
        
        #read in all ACE files in data directory
        plms_df = pd.read_table(f_aces,engine='c',header=0,delim_whitespace=True,skiprows=38) 
        #drop bogus first row
        plms_df.drop(plms_df.index[0],inplace=True)
        
        #get the speed of the wind
        plms_df['SPEED'] = plms_df['proton_speed'] # np.sqrt(np.power(plms_df.Vx.values,2)+np.power(plms_df.Vy.values,2)+np.power(plms_df.Vz,2))
        plms_df['Np'] = plms_df['proton_density']
        plms_df['Tth'] = plms_df['proton_temp']
        plms_df['day'] = plms_df['day'].astype('int').map('{:03d}'.format)
        plms_df['hour'] = plms_df['hr'].astype('int').map('{:02d}'.format)
        plms_df['minute'] = plms_df['min'].astype('int').map('{:02d}'.format)
        plms_df['second'] = plms_df['sec'].astype('int').map('{:02d}'.format)
        
        
        #remove fill values
        p_den = plms_df.Np > -9990.
        p_tth = plms_df.Tth > -9990.
        p_spd = plms_df.SPEED > -9990.
        
        plms_df = plms_df[((p_den) & (p_tth) & (p_spd))]

        amu = 1.660538921e-27#kg/amu
        mp  = 1.00727647*amu #proton mass to kg
        kb  = 1.38047e-29  #boltzmann's constant in units kg/(km/s)^2/K
        
        #make Vth value
        plms_df['Vth'] = np.sqrt(2.*kb*plms_df.Tth/(mp)) #k in units of proton mass*(km/s)^2/K
        
        #convert columns to datetime column
        plms_df['time_dt'] = pd.to_datetime(plms_df['year'].astype('str')+plms_df['day'].astype('str')+plms_df['hour']+plms_df['minute']+plms_df['second'],format='%Y%j%H%M%S')
        plms_df['time_str'] = plms_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
        #set index to be time
        plms_df.set_index(plms_df['time_dt'],inplace=True)
        #fix index not monotonic
        plms_df.sort_index(inplace=True) #.reindex(newIndex.sort_values(), method='ffill')

    #read in wind data
    elif k == 'wind':
        #find all Wind files in data directory
        f_wind = '../wind/data/wind_swe_2m_sw2016.asc'
        
        wind_nm = ['year','day','flag','SPEED','u_SPEED','Vx','u_Vx','Vy','u_Vy',
                   'Vz','u_Vz','Vth','u_Vth','Np','u_Np','Bx','By','Bz','X','Y','Z']
        wind_cl = [0,1,2,3,4,5,6,7,8,9,10,11,12,21,22,48,49,50,53,54,55]
         
        
        
        #read in all Wind files in data directory
        plms_df = pd.read_table(f_wind,engine='c',names=wind_nm,
                                usecols=wind_cl,delim_whitespace=True) 
        
        #remove fill values
        p_flag = plms_df.flag > 1.
        #remove fill values
        p_den = plms_df.Np < 9990.
        p_vth = plms_df.Vth < 9990.
        p_spd = plms_df.SPEED < 9990.
        
        plms_df = plms_df[((p_flag) & (p_den) & (p_vth) & (p_spd)) ]
        
        
        amu = 1.660538921e-27#kg/amu
        mp  = 1.00727647*amu #proton mass to kg
        kb  = 1.38047e-29  #boltzmann's constant in units kg/(km/s)^2/K
        
        
        #convert columns to datetime column
        dt = [timedelta(days=i)+datetime(2015,12,31,0,0,0) for i in plms_df.day]
        plms_df['time_dt'] =  dt

    elif k == 'dscovr':
        #find all DSCOVR files in data directory
        f_dscovr = '../dscovr/data/dscovr_h1_fc_2016.txt'
        
        #read in all DSCOVR files in data directory
        plms_df = pd.read_table(f_dscovr,engine='python',delim_whitespace=True)
        #convert columns to datetime column
        plms_df['time_dt'] = pd.to_datetime(plms_df['YEAR'].map('{:02}'.format)+':'+plms_df['MO'].map('{:02}'.format)+plms_df['DD'].map('{:02}'.format)+
                                              plms_df['HR'].map('{:02}'.format)+plms_df['MN'].map('{:02}'.format)+plms_df['SC'].map('{:02}'.format),format='%Y:%m%d%H%M%S')
        
        #smooth to 1 minutes to removed small scale variation
        plms_df['SPEED'] = np.sqrt(plms_df.Vx**2+plms_df.Vy**2+plms_df.Vz**2)
        #remove fill values
        p_den = plms_df.Np < 9990.
        p_vth = plms_df.Vth < 9990.
        p_spd = plms_df.SPEED < 9990.
        
        plms_df = plms_df[((p_flag) & (p_den) & (p_vth) & (p_spd)) ]
        
        
        plms_df['time_str'] = plms_df['time_dt'].dt.strftime('%Y/%m/%dT%H:%M:%S')
        #set index to be time
        plms_df.set_index(plms_df['time_dt'],inplace=True)
                
        

    #set index to be time
    plms_df.set_index(plms_df['time_dt'],inplace=True)
    
    
    
    
    plms_df['shock'] = 0
    #locate shocks and update parameter to 1
    for i in shock_times.start_time_dt:
        #less than 120s seconds away from shock claim as part of shock (output in nano seconds by default)
        shock, = np.where(np.abs(((plms_df.time_dt-i).values/1.e9).astype('float')) < 70.)
        plms_df['shock'][shock] = 1
    
    
    
    
    span = '3600s'
    plms_df = format_df(plms_df,span=span)
    
    #columns to use for training and secondary model
    trn_cols = ['sig_speed','sig_Np','sig_Vth','intercept']
    #columns to use in the model
    use_cols = ['Np_abs_power','speed_abs_power','Npth_abs_power' ,'Vth_abs_power','intercept']
    
    
    #cut non finite power values
    plms_df = plms_df[np.isfinite(plms_df.power)]
    

    #do training on first input using sigma of events
    if k == trainer:
  
        #set up sampling over sigma events
        samp = 10
        sig_cnts = np.zeros(samp)
        p_ran = np.linspace(3,8,samp)

        #settle on 5 sigma events in Wind
        p_ran = [4.5]
   
 
        #create dictionaries for sigma training level
        p_dct = {}

        #prediction for logit model as a fucntion of power and sigma training
        log_m = {}
        #log_p = {}
        #log_s = {}

        #for i in use_cols: p_dct[i] = np.percentile(plms_df[i],p_ran)
        for i in trn_cols: p_dct[i] = p_ran
        
        #locate shocks and update parameter to 1
        for j,i in enumerate(p_ran):
            #shock name with sigma values
            var = 'shock_{0:3.2f}'.format(i).replace('.','')
        
            #keep n sigma events for bokeh plots
            if j == 0: p_var = var
        
            #Create variable where identifies shock
            plms_df[var] = 0
        
            #loop over variable to drived logic for fitting
            log_test = [False]*len(plms_df.index)
            for p in trn_cols: log_test = ((log_test) | (plms_df[p] > p_dct[p][j]))
        
            #Train that all events with a jump greater than n sigma initially marked as shocks
            plms_df[var][log_test] = 1
            
            #build rough preliminary shock model based on observations
            try:
                #First use a power model
                logit_p = sm.Logit(plms_df[var],plms_df[use_cols])
                sh_rs_p = logit_p.fit()                                         #divide to get into units of seconds
                #next use the local vairation model
                logit_s = sm.Logit(plms_df[var],plms_df[trn_cols])
                sh_rs_s = logit_s.fit()                                         #divide to get into units of seconds

                #store in model array
                log_m['predict_power_'+var] = sh_rs_p
                log_m['predict_sigma_'+var] = sh_rs_s
            except:
                sig_cnts[j] = np.nan
                continue
        
            #get predictions for full set
            plms_df['predict_power_'+var] = sh_rs_p.predict(plms_df[use_cols])
            plms_df['predict_sigma_'+var] = sh_rs_s.predict(plms_df[trn_cols])
            plms_df['predict_'+var] = plms_df['predict_power_'+var].values*plms_df['predict_sigma_'+var].values
                                      
            events, = np.where(plms_df['predict_'+var] > 0.990)
            if events.size > 0: sig_cnts[j] = events.size
        #use the training model on all other space craft
    else:
        for j,i in enumerate(p_ran):
            try:
                var = 'shock_{0:3.2f}'.format(i).replace('.','')
                plms_df['predict_power_'+var] = log_m['predict_power_'+var].predict(plms_df[use_cols])
                plms_df['predict_sigma_'+var] = log_m['predict_sigma_'+var].predict(plms_df[trn_cols])
                plms_df['predict_'+var] = plms_df['predict_power_'+var].values*plms_df['predict_sigma_'+var].values
            except KeyError:
                continue
    
    
    #save output
    plms_df.to_pickle('../{0}/data/y2016_power_formatted.pic'.format(k))
    
    
    
    #Do parameter calculation for all previous years 
    #calculate difference in parameters
    if full_soho:
        full_df = format_df(full_df,span=span)
    
    
    #get predictions for the Mission long CELIAS mission
    if full_soho: 
        full_df['predict'] = sh_rs.predict(full_df[use_cols])
        best_df = full_df[full_df.predict >= 0.90]
        best_df.to_csv('../soho/archive_shocks.csv',sep=';')
    
    
    #create figure object
    ####fig,ax = plt.subplots(figsize=(12,7))
    ####
    #####plot solar wind speed
    ####ax.scatter(p_ran,sig_cnts,color='black')
    ####ax.set_xlabel(r'Event Percentile n$\sigma$')
    ####ax.set_ylabel(r'\# of Events (2016)')
    ####ax.set_yscale('log')
    ####ax.set_ylim([.5,2E6])
    ####fancy_plot(ax)
    ####
    ####fig.savefig('../plots/{0}_num_events_power_cut.png'.format(k),bbox_inches='tight',bbox_pad=0.1)
    ####fig.savefig('../plots/{0}_num_events_power_cut.eps'.format(k),bbox_inches='tight',bbox_pad=0.1)
    
    
    
    ###########
    #nBOKEH PLOT
    #For the training set
    ###########
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.plotting import figure, show,save
    from bokeh.layouts import column,gridplot
    
    
    ##########################################
    #Create parameters for comparing data sets
    ##########################################
    if create_bokeh:
        source = ColumnDataSource(data=plms_df[(((plms_df["predict_sigma_{0}".format(p_var)] > .1) & ((plms_df["predict_power_{0}".format(p_var)] > .1))) | (plms_df.shock == 1))])
        tools = "pan,wheel_zoom,box_select,reset,hover,save,box_zoom"
        
        tool_tips = [("Date","@time_str"),
                     ("Del. Np","@sig_Np"),
                     ("Del. Speed","@sig_speed"),
                     ("Del. Vth","@sig_Vth"),
                     ("Predict Sigma","@predict_sigma_{0}".format(p_var)),
                     ("Predict Power","@predict_power_{0}".format(p_var)),
                     ("Predict Total","@predict_{0}".format(p_var)),
                     ("Power","@power"),
                     ]
        

        #figure title
        fig_title = '{0} Discontinuities'.format(k.upper())
        
        p1 = figure(title=fig_title,tools=tools)
        p1.scatter('sig_Np','sig_Vth',color='black',source=source)
        p1.select_one(HoverTool).tooltips = tool_tips
        p1.xaxis.axis_label = 'Delta Np/sig(Np)'
        p1.yaxis.axis_label = 'Delta Vth/sig(Vth)'
                                           
        p2 = figure(title=fig_title,tools=tools)
        p2.scatter('sig_Vth','sig_speed',color='black',source=source)
        p2.select_one(HoverTool).tooltips = tool_tips
        p2.xaxis.axis_label = 'Delta Vt/sig(Vt)'
        p2.yaxis.axis_label = 'Delta |V|/sig(V)'
                                           
        p3 = figure(title=fig_title,tools=tools)
        p3.scatter('sig_speed','sig_Np',color='black',source=source)
        p3.select_one(HoverTool).tooltips = tool_tips
        p3.xaxis.axis_label = 'Delta |V|/sig(V)'
        p3.yaxis.axis_label = 'Delta Np/sig(Np)'
                                           
        p4 = figure(title=fig_title,tools=tools)
        p4.scatter('shock','predict_{0}'.format(p_var),color='black',source=source)
        p4.select_one(HoverTool).tooltips = tool_tips
        p4.xaxis.axis_label = '{0} DB SHOCK'
        p4.yaxis.axis_label = 'My Shock'
                                           
        p5 = figure(title=fig_title,tools=tools)
        p5.scatter('SPEED','Np',color='black',source=source)
        p5.select_one(HoverTool).tooltips = tool_tips
        p5.xaxis.axis_label = '|V| [km/s]'
        p5.yaxis.axis_label = 'Np [cm^-3]'
    
        p6 = figure(title=fig_title,tools=tools)
        p6.scatter('SPEED','Vth',color='black',source=source)
        p6.select_one(HoverTool).tooltips = tool_tips
        p6.xaxis.axis_label = '|V| [km/s]'
        p6.yaxis.axis_label = 'Vth [km/s]'
                                           
        p7 = figure(title=fig_title,tools=tools)
        p7.scatter('Np','Vth',color='black',source=source)
        p7.select_one(HoverTool).tooltips = tool_tips
        p7.xaxis.axis_label = 'Np [cm^-3]'
        p7.yaxis.axis_label = 'Vth [km/s]'
    
        p8 = figure(title=fig_title,tools=tools)
        p8.scatter('Np_power','speed_power',color='black',source=source)
        p8.select_one(HoverTool).tooltips = tool_tips
        p8.xaxis.axis_label = 'Np Power [~J/s]'
        p8.yaxis.axis_label = 'Speed Power [~J/s]'
    
        p11 = figure(title=fig_title,tools=tools)
        p11.scatter('Npth_power','Vth_power',color='black',source=source)
        p11.select_one(HoverTool).tooltips = tool_tips
        p11.xaxis.axis_label = 'Np Th Power [~J/s]'
        p11.yaxis.axis_label = 'Vth Power [~J/s]'
    
        p9 = figure(title=fig_title,tools=tools)
        p9.scatter('time_dt','shock'.format(p_var),color='black',source=source)
        p9.select_one(HoverTool).tooltips = tool_tips
        p9.yaxis.axis_label = '{0} DB SHOCK'
        p9.xaxis.axis_label = 'Time'
                                           
    
        p10 = figure(title=fig_title,tools=tools)
        p10.scatter('time_dt','predict_{0}'.format(p_var),color='black',source=source)
        p10.select_one(HoverTool).tooltips = tool_tips
        p10.yaxis.axis_label = 'Predict SHOCK'
        p10.xaxis.axis_label = 'Time'
    
        save(gridplot([p1,p2],[p3,p4],[p5,p6],[p7,p8],[p9,p10],[p11]),filename='../plots/bokeh_power_training_plot_{0}.html'.format(k))
        
