import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot

#format input data frame
def format_df(inpt_df,p_var,span='3600s',center=True):

    '''
    format_df is a function which formats an input data frame with plasma parameters
    into a format


    '''

    #range check for variables
    inpt_df.SPEED[((inpt_df.SPEED > 2000) | (inpt_df.SPEED < 200))] = -9999.0
    inpt_df.Vth[((inpt_df.Vth > 1E2) | (inpt_df.Vth < 0))] = -9999.0
    inpt_df.Np[((inpt_df.Np > 1E4) | (inpt_df.Np < 0))] = -9999.0
    inpt_df.Bx[np.abs(inpt_df.Bx) > 1E3] = -9999.0
    inpt_df.By[np.abs(inpt_df.By) > 1E3] = -9999.0
    inpt_df.Bz[np.abs(inpt_df.Bz) > 1E3] = -9999.0

    #output data frame
    #outp_df = inpt_df.copy()
    
    
    #check quality 
    p_den = inpt_df.Np > -9990.
    p_vth = inpt_df.Vth > -9990.
    p_spd = inpt_df.SPEED > -9990.
    p_bfx = inpt_df.Bx > -9990.
    p_bfy = inpt_df.By > -9990.
    p_bfz = inpt_df.Bz > -9990.

    #only keep times with good data in plasma or magnetic field
    plsm_df = inpt_df[((p_den) & (p_vth) & (p_spd))]
    magf_df = inpt_df[((p_bfx) & (p_bfy) & (p_bfz))]
    
    #Only fill for ACE
    #if k == 'ace':
    #    #replace bad values with nans and pervious observation fill previous value
    #   parameters = ['SPEED','Vth','Np','Bx','By','Bz'] 
    #   for p in parameters:
    #        inpt_df.loc[inpt_df[p] < -9990.,p] = np.nan
    #        inpt_df[p].ffill(inplace=True)
    #

    #Do parameter calculation of different cadences
    magf_df['del_time_mag'] = np.abs(mag[k]['time_dt_mag'].diff(1).values.astype('double')/1.e9)
    plsm_df['del_time_pls'] = np.abs(pls[k]['time_dt_pls'].diff(1).values.astype('double')/1.e9)


    #convert span to a number index so I can use the logic center = True
    #assumes format_df import is in s
    #then divide by the space craft jump time
    if center:
        span_mag = int(round(float(span[:-1])/inpt_df.del_time_mag.median()))
        span_pls = int(round(float(span[:-1])/inpt_df.del_time_pls.median()))
    #else use the runup (left) span
    else:
        span_mag = span
        span_pls = span

    #time cadence parameter to add to plasma and magnetic field time series
    par_ind_mag = inpt_df.del_time_mag.median()**2./60./inpt_df.del_time_mag
    par_ind_pls = inpt_df.del_time_pls.median()**2./60./inpt_df.del_time_pls

    #calculate difference in parameters
    plsm_df['ldel_speed'] = np.abs(plsm_df['SPEED'].diff(-1)/plsm_df.del_time_pls)
    plsm_df['ldel_Np'] = np.abs(plsm_df['Np'].diff(-1)/plsm_df.del_time_pls)
    plsm_df['ldel_Vth'] = np.abs(plsm_df['Vth'].diff(-1)/plsm_df.del_time_pls)

    #calculat plasma parameters rolling median
    plsm_df['roll_med_speed'] = plsm_df['SPEED'].rolling(span_pls,min_periods=3,center=center).median()
    plsm_df['roll_med_Np']    = plsm_df['Np'].rolling(   span_pls,min_periods=3,center=center).median()
    plsm_df['roll_med_Vth']   = plsm_df['Vth'].rolling(  span_pls,min_periods=3,center=center).median()

    #calculate difference in plasma parameters from rolling median
    plsm_df['diff_med_speed'] = plsm_df.SPEED-plsm_df.roll_med_speed
    plsm_df['diff_med_Np']    = plsm_df.Np-plsm_df.roll_med_Np
    plsm_df['diff_med_Vth']   = plsm_df.Vth-plsm_df.roll_med_Vth

    #calculate difference in plasma parameters from rolling median
    plsm_df['roll_diff_med_speed'] = plsm_df.diff_med_speed.rolling(span_pls,min_periods=3,center=center).median()
    plsm_df['roll_diff_med_Np']    = plsm_df.diff_med_Np.rolling(   span_pls,min_periods=3,center=center).median()
    plsm_df['roll_diff_med_Vth']   = plsm_df.diff_med_Vth.rolling(  span_pls,min_periods=3,center=center).median()

    #calculate acceleration in plasma parameters from rolling median
    plsm_df['accl_diff_speed'] = plsm_df.diff_med_speed-plsm_df.roll_diff_med_speed
    plsm_df['accl_diff_Np']    = plsm_df.diff_med_Np-plsm_df.roll_diff_med_Np
    plsm_df['accl_diff_Vth']   = plsm_df.diff_med_Vth-plsm_df.roll_diff_med_Vth

    #calculate sigma in plasma parameters from rollin median
    plsm_df['diff_sig_speed'] = np.sqrt((plsm_df.diff_med_speed**2.).rolling(span_pls,min_periods=3,center=center).median())
    plsm_df['diff_sig_Np']    = np.sqrt((plsm_df.diff_med_Np   **2.).rolling(span_pls,min_periods=3,center=center).median()) 
    plsm_df['diff_sig_Vth']   = np.sqrt((plsm_df.diff_med_Vth  **2.).rolling(span_pls,min_periods=3,center=center).median()) 

    #calculate acceleration in plasma parameters from rolling median
    plsm_df['accl_sig_speed'] = np.sqrt((plsm_df['accl_diff_speed']**2.).rolling(span_pls,min_periods=3,center=center).median()) 
    plsm_df['accl_sig_Np']    = np.sqrt((plsm_df['accl_diff_Np']   **2.).rolling(span_pls,min_periods=3,center=center).median()) 
    plsm_df['accl_sig_Vth']   = np.sqrt((plsm_df['accl_diff_Vth']  **2.).rolling(span_pls,min_periods=3,center=center).median()) 

    #calculate snr in plasma parameters from rollin median
    #Change to difference in sigma per minute time period 2017/10/31
    plsm_df['diff_snr_speed'] = np.abs(plsm_df.diff_med_speed)/plsm_df.diff_sig_speed *par_ind_pls
    plsm_df['diff_snr_Np']    = np.abs(plsm_df.diff_med_Np)/plsm_df.diff_sig_Np       *par_ind_pls
    plsm_df['diff_snr_Vth']   = np.abs(plsm_df.diff_med_Vth)/plsm_df.diff_sig_Vth     *par_ind_pls

    #calculate snr in plasma acceleration parameters from rollin median
    plsm_df['accl_snr_speed'] = np.abs(plsm_df.accl_sig_speed)/plsm_df.accl_sig_speed
    plsm_df['accl_snr_Np']    = np.abs(plsm_df.accl_sig_Np   )/plsm_df.accl_sig_Np   
    plsm_df['accl_snr_Vth']   = np.abs(plsm_df.accl_sig_Vth  )/plsm_df.accl_sig_Vth

    #calculate power in the diffence for paramaters
    plsm_df['diff_pow_speed'] = (np.abs(plsm_df.diff_med_speed)/plsm_df.del_time_pls)*(2.*plsm_df.roll_med_Np)
    plsm_df['diff_pow_Np']    = (np.abs(plsm_df.diff_med_Np)   /plsm_df.del_time_pls)*(plsm_df.roll_med_speed**2.+plsm_df.roll_med_Vth**2.)
    plsm_df['diff_pow_Vth']   = (np.abs(plsm_df.diff_med_Vth)  /plsm_df.del_time_pls)*(2.*plsm_df.roll_med_Np)
                             
    #calculate increase in power in the diffence for paramaters
    plsm_df['diff_acc_speed'] = (np.abs(plsm_df.diff_med_speed.diff(1))/plsm_df.del_time_pls)*(2.*plsm_df.roll_med_Np)
    plsm_df['diff_acc_Np']    = (np.abs(plsm_df.diff_med_Np.diff(1))   /plsm_df.del_time_pls)*(plsm_df.roll_med_speed**2.+plsm_df.roll_med_Vth**2.)
    plsm_df['diff_acc_Vth']   = (np.abs(plsm_df.diff_med_Vth.diff(1))  /plsm_df.del_time_pls)*(2.*plsm_df.roll_med_Np)

    #calculate B parameters rollin median
    magf_df['roll_med_Bx'] = magf_df['Bx'].rolling(span_mag,min_periods=3,center=center).median()
    magf_df['roll_med_By'] = magf_df['By'].rolling(span_mag,min_periods=3,center=center).median()
    magf_df['roll_med_Bz'] = magf_df['Bz'].rolling(span_mag,min_periods=3,center=center).median()

    #calculate difference B parameters from rollin median
    magf_df['diff_med_Bx'] = magf_df.Bx-magf_df.roll_med_Bx
    magf_df['diff_med_By'] = magf_df.By-magf_df.roll_med_By
    magf_df['diff_med_Bz'] = magf_df.Bz-magf_df.roll_med_Bz

    #calculate sigma in B parameters from rollin median
    magf_df['diff_sig_Bx'] = np.sqrt((magf_df.diff_med_Bx**2.).rolling(span_mag,min_periods=3,center=center).median())
    magf_df['diff_sig_By'] = np.sqrt((magf_df.diff_med_By**2.).rolling(span_mag,min_periods=3,center=center).median())
    magf_df['diff_sig_Bz'] = np.sqrt((magf_df.diff_med_Bz**2.).rolling(span_mag,min_periods=3,center=center).median())

    #calculate snr in B parameters from rollin median
    #Change to difference in sigma per minute time period 2017/10/31
    magf_df['diff_snr_Bx'] = np.abs(magf_df.diff_med_Bx)/magf_df.diff_sig_Bx*par_ind_mag 
    magf_df['diff_snr_By'] = np.abs(magf_df.diff_med_By)/magf_df.diff_sig_By*par_ind_mag
    magf_df['diff_snr_Bz'] = np.abs(magf_df.diff_med_Bz)/magf_df.diff_sig_Bz*par_ind_mag

    #calculate difference B parameters
    magf_df['del_Bx'] = np.abs(magf_df['Bx'].diff(1)/magf_df.del_time_mag)
    magf_df['del_By'] = np.abs(magf_df['By'].diff(1)/magf_df.del_time_mag)
    magf_df['del_Bz'] = np.abs(magf_df['Bz'].diff(1)/magf_df.del_time_mag)

    #Find difference on otherside
    plsm_df['del_speed'] = np.abs(plsm_df['SPEED'].diff(1)/plsm_df.del_time_pls)
    plsm_df['del_Np']    = np.abs(plsm_df['Np'].diff(1)/   plsm_df.del_time_pls)
    plsm_df['del_Vth']   = np.abs(plsm_df['Vth'].diff(1)/  plsm_df.del_time_pls)

    #get the Energy Change per second
    plsm_df['power'] = plsm_df.del_Np*plsm_df.SPEED**2.+2.*plsm_df.del_speed*plsm_df.Np*plsm_df.SPEED
    plsm_df['Np_power'] = plsm_df.del_Np*plsm_df.SPEED**2.
    plsm_df['speed_power'] = 2.*plsm_df.del_speed*plsm_df.Np*plsm_df.SPEED
    plsm_df['Npth_power'] = plsm_df.del_Np*plsm_df.Vth**2.
    plsm_df['Vth_power'] = 2.*plsm_df.del_Vth*plsm_df.Np*plsm_df.Vth

    #absolute value of the power
    plsm_df['abs_power'] = np.abs(plsm_df.power)
    plsm_df['Np_abs_power'] = np.abs(plsm_df.Np_power)
    plsm_df['speed_abs_power'] =  np.abs(plsm_df.speed_power)
    plsm_df['Npth_abs_power'] = np.abs(plsm_df.Npth_power)
    plsm_df['Vth_abs_power'] =  np.abs(plsm_df.Vth_power)

    #calculate variance normalized parameters
    plsm_df['std_speed'] = plsm_df.roll_med_speed/plsm_df.del_time_pls
    plsm_df['std_Np']    = plsm_df.roll_med_Np   /plsm_df.del_time_pls
    plsm_df['std_Vth']   = plsm_df.roll_med_Vth  /plsm_df.del_time_pls

    #calculate standard dev in B parameters
    magf_df['std_Bx'] = magf_df.diff_sig_Bx
    magf_df['std_By'] = magf_df.diff_sig_By
    magf_df['std_Bz'] = magf_df.diff_sig_Bz
    
    #Significance of the variation in the wind parameters
    plsm_df['sig_speed'] = plsm_df.del_speed/plsm_df.std_speed
    plsm_df['sig_Np']    = plsm_df.del_Np/plsm_df.std_Np
    plsm_df['sig_Vth']   = plsm_df.del_Vth/plsm_df.std_Vth

    #significance of variation in B parameters
    magf_df['sig_Bx'] = magf_df.del_Bx/magf_df.std_Bx
    magf_df['sig_By'] = magf_df.del_By/magf_df.std_By
    magf_df['sig_Bz'] = magf_df.del_Bz/magf_df.std_Bz
    
    #fill pesky nans and infs with 0 values
    pls_fill = ['sig_speed','sig_Np','sig_Vth', 
                'diff_snr_speed','diff_snr_Np','diff_snr_Vth',
                'diff_pow_speed','diff_pow_Np','diff_pow_Vth', 
                'diff_acc_speed','diff_acc_Np','diff_acc_Vth']  
    mag_fill = ['sig_Bx','sig_By','sig_Bz',
                'diff_snr_Bx','diff_snr_By','diff_snr_Bz']
                
                

    #loop through and replace fill values with -9999.9
    for i in pls_fill: 
        plsm_df[i].replace(np.inf,np.nan,inplace=True)
        plsm_df[i].fillna(value=-9999.9,inplace=True)

    #loop through and replace fill values with -9999.9
    for i in mag_fill: 
        magf_df[i].replace(np.inf,np.nan,inplace=True)
        magf_df[i].fillna(value=-9999.9,inplace=True)

    
    #create an array of constants that hold a place for the intercept 
    #plsm_df['intercept'] = 1 


    #only guess times with good data in plasma or magnetic field respectively
    m_var = p_var.replace('predict','predict_sigma')
    plsm_df.loc[:,p_var] = logit_model(plsm_df,-3.2437,0.2365,0.0464,0.0594)
    magf_df.loc[:,m_var] = logit_model(magf_df,-10.0575,0.6861,0.7071,0.6640)

    #cur arrays to the bar minimum for matching
    plsm_df = plsm_df.loc[:,'p_var']
    magf_df = magf_df.loc[:,'m_var']
    
    #update array with new p-values by matching on indices
    outp_df  = pd.merge(outp_df,plsm_df,how='left',left_index=True,right_index=True)
    outp_df  = pd.merge(outp_df,magf_df,how='left',left_index=True,right_index=True)


    return outp_df


#probability model and parameters to use
def logit_model(df,b0,b1,b2,b3,plasma=True):
    if plasma:
        return 1./(np.exp(-(df.diff_snr_speed*b1+df.diff_snr_Vth*b2+df.diff_snr_Np*b3+b0))+1.) 
    else:
        return 1./(np.exp(-(df.diff_snr_Bx*b1+df.diff_snr_By*b2+df.diff_snr_Bz*b3+b0))+1.) 


#function to find the Chi^2 min value given a set of parameters
def chi_min(p_mat,par,rgh_chi_t,plsm,k,ref_chi_t=pd.to_timedelta('10 minutes'),refine=True,n_fine=4,plot=False):

    """
    chi_min computes the chi^2 min. time for a given set of parameters

    p_mat: Pandas DataFrame
    Solar wind pandas DataFrame for a space craft roughly sampled

    par : list
    List of parameters to use in the Chi^2 minimization 

    rgh_chi_t: Pandas datetime delta object
    Time around a given time in p_mat to include in chi^2 minimization  

    ref_chi_t: Pandas datetime delta object
    Time around a given time in refined grid to include in chi^2 minimization  

    plsm: dict
    Dictionary of plasma DataFrames of plasma Dataframes
   
    k   : string
    Spacecraft name in plsm dictionary
   

    refine: boolean
    Whether to use a refined window (Default = True)

    n_fine : integer
    Number of chi^2 min values from the rough grid to check with the fine grid (Default = 4)

    plot  : boolean
    Plot Chi^2 minimization window (Default = False)

    RETURNS
    --------
    i_min : datetime index
    Datetime index of best fit Chi^2 time

    """

    #set up chisquared array in pandas object
    p_mat.loc[:,'chisq'] = -99999.9
    for time in p_mat.index:
        #get a region around one of the best fit times
        com_slice = [time-rgh_chi_t,time+rgh_chi_t]
        c_mat = plsm[k].loc[com_slice[0]:com_slice[1]]
    
        #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
        c_mat.index = c_mat.index+(i-time)
    
    
        #remove Speed fill values by interpolation
        c_mat.loc[c_mat.SPEED < 0.,'SPEED'] = np.nan
        c_mat.SPEED.interpolate('time',inplace=True)
    
        #get training spacecraft time range
        t_mat = plsm[trainer].loc[c_mat.index.min():c_mat.index.max()]
    
        #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid and interpolate
        c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')
    
    
        #get median offset to apply to match spacecraft
        off_speed = c_mat.SPEED.median()-t_mat.SPEED.median()
        c_mat.SPEED = c_mat.SPEED-off_speed
    

        #compute chi^2 value for Wind and other spacecraft
        p_mat.loc[time,'chisq'] = np.sqrt(np.sum(((c_mat.loc[:,par]-t_mat.loc[:,par])**2.).values))
    
    
        #create figure to check matchin
        if plot:
            ax.scatter(c_mat.index,c_mat.SPEED,label=time.to_pydatetime().strftime('%Y/%m/%dT%H:%M:%S')+' chisq = {0:4.0f}'.format(p_mat.loc[time,'chisq']))
            ax.plot(t_mat.index,t_mat.SPEED,label='',color='black')
    
    
         
    
    if plot:
        ax.legend(loc='upper left',frameon=False,scatterpoints=1)
        ax.set_xlim([t_mat.index.min(),t_mat.index.max()])
        ax.set_ylabel('Speed [km/s]')
        ax.set_xlabel('Time [UTC]')
        fancy_plot(ax)
        #ax.set_ylim([300,1000.])
        plt.show()
    
    #get the index of minimum chisq value
    i_min = p_mat['chisq'].idxmin()
    
    #use fine grid around observations to match time offset locally
    if refine:

        #Find 4 lowest chisq min times
        p_temp = p_mat.sort_values('chisq',ascending=True )[:n_fine]
 
        #get all values in top n_fine at full resolution
        p_mat  = plsm[k].loc[p_temp.index.min()-rgh_chi_t:p_temp.index.max()+rgh_chi_t]
        print(p_temp.index.min()-rgh_chi_t,p_temp.index.max()+rgh_chi_t)
 
    
        #loop over all indexes in refined time window
        for time in p_mat.index:
            #get a region around one of the best fit times
            com_slice = [time-ref_chi_t,time+ref_chi_t]
            c_mat = plsm[k].loc[com_slice[0]:com_slice[1]]
    
            #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
            c_mat.index = c_mat.index+(i-time)
    
    
            #remove Speed fill values by interpolation
            c_mat.loc[c_mat.SPEED < 0.,'SPEED'] = np.nan
            c_mat.SPEED.interpolate('time',inplace=True)
    
            #get trainint spacecraft time range
            t_mat = plsm[trainer].loc[c_mat.index.min():c_mat.index.max()]
    
            #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid and interpolate
            c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')
    
            #get median offset to apply to match spacecraft
            off_speed = c_mat.SPEED.median()-t_mat.SPEED.median()
            c_mat.SPEED = c_mat.SPEED-off_speed
    
            #compute the chisq value in SPEED from the top ten probablilty array including the median offsets
            p_mat.loc[time,'chisq'] = np.sqrt(np.sum(((c_mat.loc[:,par]-t_mat.loc[:,par])**2.).values))
    
    
        #get the index of minimum refined chisq value
        i_min = p_mat['chisq'].idxmin()

    return i_min
    
#dictionary for storing the Pandas Data frame
pls = {}
mag = {}
plsm = {}


#use space craft position to get range of time values
use_craft = False

#use best chisq to find the best time in between spacecraft
use_chisq = True

#plot chisq min procedure
plot = False

#refine chisq min with closer time grid
refine = True

#set use to use all spacecraft
craft = ['Wind','DSCOVR','ACE','SOHO']
col   = ['blue','black','red','teal']
mar   = ['D','o','s','<']
marker = {}
color  = {}


#Use five min cadence for rough chi^2 min
downsamp = '5T'


#create dictionaries for labels
for j,i in enumerate(craft):
    marker[i] = mar[j]
    color[i]  = col[j]

#space craft to use n sigma events to find other events
#change craft order to change trainer 
trainer = craft[0]


#sigma level to use from wind training
sig_l = 5.0
#use sig_l to create shock prediction variable in dataframes
p_var = 'predict_shock_{0:3.2f}'.format(sig_l).replace('.','')
#fractional p value to call an "event"
p_val = 0.980 
p_val = 0.999 

#read in all spacraft events
#for k in craft: plsm[k] = pd.read_pickle('../{0}/data/y2016_power_formatted.pic'.format(k.lower()))
arch = '../cdf/cdftotxt/'
for k in craft:

    #Read in plasma and magnetic field data from full res
    pls[k] = pd.read_table(arch+'{0}_pls_formatted.txt'.format(k.lower()),delim_whitespace=True)

    #no magnetic field data from SOHO
    if k != 'soho':
        mag[k] = pd.read_table(arch+'{0}_mag_formatted.txt'.format(k.lower()),delim_whitespace=True)

        #create datetime objects from time
        pls[k]['time_dt'] = pd.to_datetime(pls[k]['Time'])
        mag[k]['time_dt'] = pd.to_datetime(mag[k]['Time'])

        #setup index
        pls[k].set_index(pls[k].time_dt,inplace=True)
        mag[k].set_index(mag[k].time_dt,inplace=True)


        #join magnetic field and plasma dataframes
        com_df  = pd.merge(mag[k],pls[k],how='outer',left_index=True,right_index=True,suffixes=('_mag','_pls'))

        #replace NaN with previously measured value
        com_df.fillna(method='bfill',inplace=True)
        
        #get degault formating for pandas dataframe
        plsm[k] = format_df(com_df,p_var) 
    else:
        pls[k]['time_dt'] = pd.to_datetime(pls[k]['Time'])
        plsm[k] = format_df(pls[k],p_var)




#use the trainer space craft to find events
tr_events = plsm[trainer][plsm[trainer][p_var] > p_val]

#Group events by 1 per hour
tr_events['str_hour'] = tr_events.time_dt.dt.strftime('%Y%m%d%H')
#attempting to pick one event at a time J. Prchlik (2017/10/30) (Did not work as 1pm of the same day)
#tr_events = tr_events[~tr_events.duplicated(['str_hour'],keep = 'first')]
#loop over events and get best probability for event within 1 hour
tr_events['group'] = range(tr_events.index.size)
for i in tr_events.index:
    tr_events['t_off'] = abs((i-tr_events.index).total_seconds())
    match = tr_events[tr_events['t_off'] < 3600.] #anytime where the offest is less than 1 hour 
    tr_events.loc[match.index,'group'] = match.group.values[0]

#use groups to cut out duplicates
tr_events = tr_events[~tr_events.duplicated(['group'],keep = 'first')]

#get strings for times around each event#
window = {}
window['DSCOVR'] = pd.to_timedelta('180 minutes')
window['ACE'] = pd.to_timedelta('180 minutes')
window['SOHO'] = pd.to_timedelta('180 minutes')
window['Wind'] = pd.to_timedelta('180 minutes')

#define rough chi min time  to cal Chi^2 min for each time
rgh_chi_t = pd.to_timedelta('30 minutes')

#get strings for times around each event when refining chi^2 time
ref_window = {}
ref_window['DSCOVR'] = pd.to_timedelta('25 minutes')
ref_window['ACE'] = pd.to_timedelta('25 minutes')
ref_window['SOHO'] = pd.to_timedelta('25 minutes')
ref_window['Wind'] = pd.to_timedelta('25 minutes')

#refined window to calculate Chi^2 min for each time
ref_chi_t = pd.to_timedelta('10 minutes')


#space craft to match with trainer soho
craft.remove(trainer)

#use wind speed to esimate possible time delay
min_vel = 100. #km/s
max_vel = 5000. #km/s
Re = 6378. #Earth radius in km

#convert ace coordinates into Re
#plsm['ACE']['X'] = plsm['ACE'].X/Re
#plsm['ACE']['Y'] = plsm['ACE'].Y/Re
#plsm['ACE']['Z'] = plsm['ACE'].Z/Re

#correct normalization of spacecraft wind parameter Vth
#Fixed J. Prchlik 2017/11/02
##amu = 1.660538921e-27#amu
##mp  = 1.00727647*amu
##kb  = 1.3906488e-23#boltzmann's constant
##plsm['wind']['Vth']   = (plsm['wind'].Vth  *1.0e3)**2.*mp/kb/2.
##plsm['ace']['Vth']    = (plsm['ace'].Vth   *1.0e3)**2.*mp/kb/2.
##plsm['dscovr']['Vth'] = (plsm['dscovr'].Vth*1.0e3)**2.*mp/kb/2.



#set up html output Event heading

ful_hdr = '''
            <!DOCTYPE html>
            <html>
            <head>
            <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
            }
            th, td {
                padding: 5px;
            }
            th {
                text-align: left;
            }
            </style>
            </head>
            <body>
            '''

par_hdr =   '''
            <h1>
            Finding &chi;<sup>2</sup> Minimization 
            </h1>
            </br>
            </br>
            <p1>
            The program for finding the &chi;<sup>2</sup> minimum for events between spacecraft works by using a primary spacecraft to find events.
            I selected Wind because I used Wind to train the model, and the events found in Wind are consistent with events found by eye or independent spacecraft 
            algorithms. I define an event discontinuity as a time where our model indicates a {0:4.2f}% probability the solar wind parameters corresponds to a 5&sigma;
            discontinuity in a component of the magnetic field. Then I create a course grid of 5 minute samples for &plusmn; 3 hours around a event. Then I shift 
            the other 3 spacecraft onto the Wind using the course grid. Then I select a 1 hour window around the shifted Wind and other spacecraft and derive a first order
            &chi;<sup>2</sup> minimum. Using the first order &chi;<sup>2</sup> minimum, I create a fine grid 50 minutes (&plusmn; 25 minutes) around the rough &chi;<sup>2</sup> minimum
            time. The fine grid cadence is set to the maximum sampling frequency of a given spacecraft. Then the program stores the &chi;<sup>2</sup> value for 20 minutes (&plusmn; 10 minutes)
            around each fine grid point. The &chi;<sup>2</sub> reported is the minimum from the time grid.
            </p1>

            </br>
            </br>
            <h4>
            Table Summary
            </h4>
            Each table contains a row for each spacecraft with observations in the requested time period. Then each row contains 6 columns. The first column labels the spacecraft for a 
            given row. Then next column (Obs. Time [UT]) gives the time of the "event" in the reference frame of the spacecraft. Offset is the offset in minutes from the Wind observation.
            P-val (plasma) is the p-val for a 5&sigma; discontinuity at the reference Obs. time for a given spacecraft's measured plasma values. 
            P-val (mag.) is the p-val for a 5&sigma; discontinuity given at the reference Obs. time for a given spacecraft's measured magnetic field values.
            Finally, an X in Use Plasma Mod. means the minimization routine use the wind Speed to find the best match &chi;<sup>2</sup> time,
            while an empty cell denotes using the magnetic field.
    
            </br>
            </br>

            '''

tab_hdr = '''

           <table style="width=100%">
              <tr>
                  <td>
                  Spacecraft
                  </td>
                  <td>
                  Obs. Time [UT]
                  </td>
                  <td>
                  Offset [min]
                  </td>
                  <td>
                  p-val (plasma) 
                  </td>
                  <td>
                  p-val (mag.) 
                  </td>
                  <td>
                  Used Plasma Mod.
                  </td>
              </tr>
                  '''

#data format for new row
new_row =   '''<tr>
                  <td>
                  {0}
                  </td>
                  <td>
                  {1:%Y/%m/%d %H:%M:%S}
                  </td>
                  <td>
                  {2:5.2f}
                  </td>
                  <td>
                  {3:4.3f}
                  </td>
                  <td>
                  {4:4.3f}
                  </td>
                  <td>
                  {5}
                  </td>
              </tr>'''
footer = '''</table>

         </br>
         </br>
         </br>


         '''
ful_ftr = '''
            </body>
            </html>'''


# write header for html page
out_f = open('../html_files/{0}_level_{1:4.0f}_full_res.html'.format(trainer.lower(),p_val*1000.).replace(' ','0'),'w')
out_f.write(ful_hdr+par_hdr.format(p_val*100.))

#get event slices 
for i in tr_events.index:
    print('########################################')
    print('NEW EVENT')
    print(trainer)
    print('{0:%Y/%m/%d %H:%M:%S}, p (plasma)={1:4.3f}, p (mag.) = {2:4.3f}'.format(i,tr_events.loc[i,p_var],tr_events.loc[i,p_var.replace('predict','predict_sigma')]))
    #get time slice around event
   
    #create table to output html table and link to github png files (https://cdn.rawgit.com/jprchlik/solar_wind_jets/4cf1c6e7)
    out_f.write(r'''<b><a href="../plots/spacecraft_events/full_res_event_{0:%Y%m%d_%H%M%S}_bigg.png"> Event on {0:%Y/%m/%d %H:%M:%S} UT (6 Hour)</a> </b>     '''.format(i))
    out_f.write(r'''<b><a href="../plots/spacecraft_events/full_res_event_{0:%Y%m%d_%H%M%S}_zoom.png"> Event on {0:%Y/%m/%d %H:%M:%S} UT (50 Min.)</a> </b>'''.format(i))
    out_f.write(tab_hdr)
    #write trainer spacecraft event
    out_f.write(new_row.format(trainer,i,0.00,tr_events.loc[i,p_var],tr_events.loc[i,p_var.replace('predict','predict_sigma')],'X'))


    #create figure showing 
    bfig, fax = plt.subplots(nrows=3,ncols=2,sharex=True,figsize=(18,18))
    bfig.subplots_adjust(hspace=0.001)
    bfig.suptitle('Event on {0:%Y/%m/%d %H:%M:%S} UT'.format(i),fontsize=24)


    #loop over all other space craft
    for k in craft:


        #initial time slice to get the approximate spacecraft differences
        init_slice = [i-window[k],i+window[k]] 
        p_mat = plsm[k].loc[init_slice[0]:init_slice[1]] 

        #print current spacecraft
        print(k)
        #use spacecraft position to get solar wind times for matching
        if use_craft:
  
            #find the median distance
            #med_dis = np.median(np.sqrt((p_mat.X-i.X)**2.+(p_mat.Y-i.Y)**2.+(p_mat.Z-i.Z)**2.))
            #Just use X distance since most of wind is in that direction
            #towards the sun is positive X (GSE)
            #use negative sign to change to increasing away from the sun
            med_dis_x = (p_mat.X-tr_events.loc[i,'X']).median()
            med_dis_y = (p_mat.Y-tr_events.loc[i,'Y']).median()
            med_dis_z = (p_mat.Z-tr_events.loc[i,'Z']).median()


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
            if p_mat.size > 0:
               #get the index of the maximum value
               i_max = p_mat[p_var].idxmax() 
               print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max={1:4.3f}'.format((i_max-i).total_seconds()/60.,p_mat.loc[i_max][p_var],i_max))
            else:
               print('No Plasma Observations')
  
        #use chisq minimum of top events in 2 hour window
        elif use_chisq:
     

            #downsample to 5 minutes for time matching in chisq
            p_mat_t = p_mat.resample(downsamp).median()

            #downsample mag to 5 minutes for time matching in chisq
            p_mag_t = p_mat_t

            #mag tolerance for using magnetometer data to match events rather than plasma parameters
            mag_tol = 0.5
            


            if (((p_mat_t.size > 0) & (p_mat_t[p_var].max() > mag_tol)) | ((k.lower() == 'soho') & (p_mat_t.size > 0.))):

                i_min = chi_min(p_mat_t,['SPEED'],rgh_chi_t,plsm,k,ref_chi_t=ref_chi_t,refine=True,n_fine=4,plot=False)
                #create figure to test fit
                if plot:
                    fig, ax = plt.subplots()
                    ax.set_title(k)



                #use full array for index matching
                p_mat = plsm[k]
                try:
 
                    if k.lower() == 'soho':
                        print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min))
                        out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],0.000,'X'))
                    else: 
                        print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
                        out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],p_mat.loc[i_min][p_var.replace('predict','predict_sigma')],'X'))
                except KeyError:
                    print('Missing Index')
            #else no plasma observations                                                                                                                                                      
            elif (((p_mat_t.size == 0) | (p_mat_t[p_var].max() <= mag_tol)) & (p_mag_t.size > 0.) & (k.lower() != 'soho')):
                print 'Using Magnetic field observations'
                #sort the cut window and get the top 10 events
                p_mat_t = p_mag_t
       
                i_min = chi_min(p_mat_t,['Bx','By','Bz'],rgh_chi_t,plsm,k,ref_chi_t=ref_chi_t,refine=True,n_fine=4,plot=False)

                #use full array for index matching
                p_mat = plsm[k]
                try:
                #print output to terminal
                    print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
                    out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],p_mat.loc[i_min][p_var.replace('predict','predict_sigma')],''))
                except KeyError:
                    print('Missing Index')
       
            else:
               print('No Plasma or Mag. Observations')
               continue

        #get a region around one of the best fit times
        plt_slice = [i_min-window[k],i_min+window[k]]
        b_mat = plsm[k].loc[plt_slice[0]:plt_slice[1]]
        
        #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
        b_mat.index = b_mat.index+(i-i_min)

        #plot plasma parameters
        #fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k.upper())         
        fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k)         
        fax[1,0].scatter(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[k],color=color[k])         
        fax[2,0].scatter(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,marker=marker[k],color=color[k])         

        fax[0,0].plot(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,color=color[k],linewidth=2,label='')         
        fax[1,0].plot(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,color=color[k],linewidth=2)         
        fax[2,0].plot(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,color=color[k],linewidth=2)         


        #plot mag. parameters
        if k.lower() != 'soho':
            fax[0,1].scatter(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,marker=marker[k],color=color[k])         
            fax[1,1].scatter(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,marker=marker[k],color=color[k])         
            fax[2,1].scatter(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,marker=marker[k],color=color[k])         

            fax[0,1].plot(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,color=color[k],linewidth=2)         
            fax[1,1].plot(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,color=color[k],linewidth=2)         
            fax[2,1].plot(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,color=color[k],linewidth=2)         

        #print separater 
        print('########################################')
    
    #get training spacecraft time range
    plt_slice = [i-window[k],i+window[k]]
    t_mat = plsm[trainer].loc[plt_slice[0]:plt_slice[1]]

    #plot plasma parameters
    fax[0,0].scatter(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,marker=marker[trainer],color=color[trainer],label=trainer.upper())         
    fax[1,0].scatter(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[trainer],color=color[trainer])         
    fax[2,0].scatter(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,marker=marker[trainer],color=color[trainer])         

    fax[0,0].plot(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,color=color[trainer],linewidth=2,label='')         
    fax[1,0].plot(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,color=color[trainer],linewidth=2)         
    fax[2,0].plot(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,color=color[trainer],linewidth=2)         
    #plot mag. parameters
    fax[0,1].scatter(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,marker=marker[trainer],color=color[trainer])         
    fax[1,1].scatter(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,marker=marker[trainer],color=color[trainer])         
    fax[2,1].scatter(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,marker=marker[trainer],color=color[trainer])         

    fax[0,1].plot(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,color=color[trainer],linewidth=2)         
    fax[1,1].plot(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,color=color[trainer],linewidth=2)         
    fax[2,1].plot(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,color=color[trainer],linewidth=2)         


    #plot observed break time
    for pax in fax.ravel():
        xoff = pd.to_timedelta('90 seconds')  
        pax.axvline(i,linewidth=3,alpha=0.5,color='purple')
        pax.axvline(i+xoff,alpha=0.5,linestyle='--',linewidth=3,color='purple')
        pax.axvline(i-xoff,alpha=0.5,linestyle='--',linewidth=3,color='purple')
       
    #plot best time for each spacecraft
    fax[0,0].legend(loc='upper left',frameon=False,scatterpoints=1)

    #plot an hour around observation
    fax[0,0].set_xlim([i-pd.to_timedelta('25 minutes'),i+pd.to_timedelta('25 minutes')])
  
    fax[0,0].set_ylabel('Np [cm$^{-3}$]',fontsize=20)
    fax[1,0].set_ylabel('Th. Speed [km/s]',fontsize=20)
    fax[2,0].set_ylabel('Flow Speed [km/s]',fontsize=20)
    fax[2,0].set_xlabel('Time [UTC]',fontsize=20)

    fax[0,1].set_ylabel('Bx [nT]',fontsize=20)
    fax[1,1].set_ylabel('By [nT]',fontsize=20)
    fax[2,1].set_ylabel('Bz [nT]',fontsize=20)
    fax[2,1].set_xlabel('Time [UTC]',fontsize=20)

    #make fancy plot
    for pax in fax.ravel(): fancy_plot(pax)
    #ax.set_ylim([300,1000.])
    bfig.savefig('../plots/spacecraft_events/full_res_event_{0:%Y%m%d_%H%M%S}_zoom.png'.format(i.to_pydatetime()),bbox_pad=.1,bbox_inches='tight')

    fax[0,0].set_xlim([i-pd.to_timedelta('180 minutes'),i+pd.to_timedelta('180 minutes')])
    bfig.savefig('../plots/spacecraft_events/full_res_event_{0:%Y%m%d_%H%M%S}_bigg.png'.format(i.to_pydatetime()),bbox_pad=.1,bbox_inches='tight')

    #close output file
    out_f.write(footer)

#write footer and close file

out_f.write(ful_ftr)
out_f.close()



