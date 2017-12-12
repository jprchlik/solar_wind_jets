import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from datetime import datetime
from multiprocessing import Pool,Lock,current_process
from functools import partial
import os
import threading
import sys
import time

from scipy.stats.mstats import theilslopes

#prevents issues with core affinity (i.e. issues with using all the processes)
#os.system("taskset -p 0xff %d" % os.getpid())

#Function to read in spacecraft
def read_in(k,p_var='predict_shock_500',arch='../cdf/cdftotxt/',mag_fmt='{0}_mag_2015_2017_formatted.txt',pls_fmt='{0}_pls_2015_2017_formatted.txt',
            start_t='2016/12/01',end_t='2017/09/24',center=False):
    """
    A function to read in text files for a given spacecraft

    Parameters
    ----------
    k: string
        Then name of a spacecraft so to format for file read in
    arch: string, optional
        The archive location for the text file to read in (Default = '../cdf/cdftotxt/')
    mag_fmt: string, optional
        The file format for the magnetic field observations (Default = '{0}_mag_formatted.txt',
        where 0 is the formatted k).
    pls_fmt: string, optional
        The file format for the plasma observations (Default = '{0}_pls_formatted.txt',
        where 0 is the formatted k).
    center = boolean, optional
        Whether the analyzed point to be center focused (center = True) or right focus (Default = False).
        Using a right focused model I find better agreement with event found by eye.
    start_t: string, optional
        Date in YYYY/MM/DD format to start looking for events (Default = '2016/06/04')
    start_t: string, optional
        Date in YYYY/MM/DD format to stop looking for events (inclusive, Default = '2017/07/31')

    Returns
    -------
    plsm: Pandas DataFrame
        A pandas dataframe with probability values and combined mag and plasma observations.
    
    """
    #Read in plasma and magnetic field data from full res
    pls = pd.read_table(arch+pls_fmt.format(k.lower()),delim_whitespace=True)

    #no magnetic field data from SOHO
    if k.lower() != 'soho':
        mag = pd.read_table(arch+mag_fmt.format(k.lower()),delim_whitespace=True)

        #create datetime objects from time
        pls['time_dt_pls'] = pd.to_datetime(pls['Time'])
        mag['time_dt_mag'] = pd.to_datetime(mag['Time'])

        #setup index
        pls.set_index(pls.time_dt_pls,inplace=True)
        mag.set_index(mag.time_dt_mag,inplace=True)

        #cut for testing reasons
        pls = pls[start_t:end_t]
        mag = mag[start_t:end_t]
        #pls = pls['2016/07/18':'2016/07/21']
        #mag = mag['2016/07/18':'2016/07/21']
        #pls = pls['2017/01/25':'2017/01/27']
        #mag = mag['2017/01/25':'2017/01/27']

        #join magnetic field and plasma dataframes
        com_df  = pd.merge(mag,pls,how='outer',left_index=True,right_index=True,suffixes=('_mag','_pls'),sort=True)

        #make sure data columns are numeric
        cols = ['SPEED','Np','Vth','Bx','By','Bz']
        com_df[cols] = com_df[cols].apply(pd.to_numeric, errors='coerce')

        #add Time string
        com_df['Time'] = com_df.index.to_datetime().strftime('%Y/%m/%dT%H:%M:%S')

        #replace NaN with previously measured value
        #com_df.fillna(method='bfill',inplace=True)
        
        #get degault formating for pandas dataframe
        plsm = format_df(com_df,p_var,center=False) 
    else:
        #work around for no Mag data in SOHO
        pls.loc[:,['Bx','By','Bz']] = 0.0
        pls['time_dt_pls'] = pd.to_datetime(pls['Time'])
        pls['time_dt_mag'] = pd.to_datetime(pls['Time'])
        pls.set_index(pls.time_dt_pls,inplace=True)
        plsm = format_df(pls,p_var,center=center)
        plsm.loc[:,['Bx','By','Bz']] = -9999.0

    #for rekeying later
    plsm['craft'] = k

    return plsm


#format input data frame
def format_df(inpt_df,p_var,span='3600s',center=False):

    '''
    format_df is a function which formats an input data frame with plasma parameters
    into a format to be used by the rest of the program. Cheifly, it determines the 
    p-values for an event.

    Parameters
    ----------
    inpt_df: Pandas DataFrame
        A combined plasma and magnetic field pandas DataFrame.
    p_var  : string 
        Formatted string of column name which will contain the trained 
        sigma levels probability level for training (i.e. 5 sigma).
    span: string, optional
        String with pandas to_timedelta like formating (Default = 3600s)
    center: boolean, optional
        Boolean specifying with probabilities should compute with point 
        centered values (center = True) or left sided values (Default = False). 

    Returns
    -------
    outp_df: Pandas DataFrame
        A dataframe with probability values populated based on a 5 sigma
        trained magnetic field model.

    '''


    #Add Mag field Magnitude
    inpt_df['Bt'] = np.sqrt(inpt_df.Bx**2.+inpt_df.By**2.+inpt_df.Bz**2.)
   

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
    p_den = ((inpt_df.Np > -9990.)    & (np.isfinite(inpt_df.Np)))
    p_vth = ((inpt_df.Vth > -9990.)   & (np.isfinite(inpt_df.Vth)))
    p_spd = ((inpt_df.SPEED > -9990.) & (np.isfinite(inpt_df.SPEED)))
    p_ptm = inpt_df.time_dt_pls.notnull()

    p_bfx = ((inpt_df.Bx > -9990.)    & (np.isfinite(inpt_df.Bx)))
    p_bfy = ((inpt_df.By > -9990.)    & (np.isfinite(inpt_df.By)))
    p_bfz = ((inpt_df.Bz > -9990.)    & (np.isfinite(inpt_df.Bz)))
    p_mtm = inpt_df.time_dt_mag.notnull()

    #only keep times with good data in plasma or magnetic field
    plsm_df = inpt_df[((p_den) & (p_vth) & (p_spd) & (p_ptm))]
    magf_df = inpt_df[((p_bfx) & (p_bfy) & (p_bfz) & (p_mtm))]

    #Do a quick spike rejection
    #calculat plasma parameters rolling median for spike rejection
    plsm_df['roll_med_spike_speed'] = plsm_df['SPEED'].rolling(60,min_periods=3,center=True).median()
    plsm_df['roll_med_spike_Np']    = plsm_df['Np'].rolling(   60,min_periods=3,center=True).median()
    plsm_df['roll_med_spike_Vth']   = plsm_df['Vth'].rolling(  60,min_periods=3,center=True).median()

    #calculate difference in plasma parameters from rolling median for spike rejection
    plsm_df['diff_med_spike_speed'] = plsm_df.SPEED-plsm_df.roll_med_spike_speed
    plsm_df['diff_med_spike_Np']    = plsm_df.Np   -plsm_df.roll_med_spike_Np
    plsm_df['diff_med_spike_Vth']   = plsm_df.Vth  -plsm_df.roll_med_spike_Vth

    #calculate sigma in plasma parameters from rollin median for spike rejection
    plsm_df['diff_sig_spike_speed'] = np.sqrt((plsm_df.diff_med_spike_speed**2.).rolling(60,min_periods=3,center=True).median())
    plsm_df['diff_sig_spike_Np']    = np.sqrt((plsm_df.diff_med_spike_Np   **2.).rolling(60,min_periods=3,center=True).median()) 
    plsm_df['diff_sig_spike_Vth']   = np.sqrt((plsm_df.diff_med_spike_Vth  **2.).rolling(60,min_periods=3,center=True).median()) 


    #difference for spike rejection
    plsm_df['diff_snr_spike_speed'] = np.abs(plsm_df.diff_med_spike_speed)/plsm_df.diff_sig_spike_speed 
    plsm_df['diff_snr_spike_Np']    = np.abs(plsm_df.diff_med_spike_Np)   /plsm_df.diff_sig_spike_Np       
    plsm_df['diff_snr_spike_Vth']   = np.abs(plsm_df.diff_med_spike_Vth)  /plsm_df.diff_sig_spike_Vth     

    #shift back
    plsm_df['l_diff_snr_spike_speed'] = plsm_df['diff_snr_spike_speed'].shift(-1) 
    plsm_df['l_diff_snr_spike_Np']    = plsm_df['diff_snr_spike_Np']   .shift(-1) 
    plsm_df['l_diff_snr_spike_Vth']   = plsm_df['diff_snr_spike_Vth']  .shift(-1) 

    #shift forward
    plsm_df['r_diff_snr_spike_speed'] = plsm_df['diff_snr_spike_speed'].shift(1) 
    plsm_df['r_diff_snr_spike_Np']    = plsm_df['diff_snr_spike_Np']   .shift(1) 
    plsm_df['r_diff_snr_spike_Vth']   = plsm_df['diff_snr_spike_Vth']  .shift(1) 

    #Sigma tolerance 
    sig_tol = 5.0

    #loop over toleranaces and fill with -9999.0 where you find an isolated spike
    tol_loop = ['speed','Np','Vth'] 
    for i in tol_loop:
        before = plsm_df['l_diff_snr_spike_'+i] < sig_tol
        during = plsm_df['diff_snr_spike_'+i]   > sig_tol
        after  = plsm_df['r_diff_snr_spike_'+i] < sig_tol

        #correct for bad variable naming
        if i == 'speed': i = i.upper()
        plsm_df.loc[((before) & (during) & (after)),i] = np.nan
        #Interpolate over filled values
        plsm_df[i].interpolate(inplace=True)





    
    #Only fill for ACE
    #if k == 'ace':
    #    #replace bad values with nans and pervious observation fill previous value
    #   parameters = ['SPEED','Vth','Np','Bx','By','Bz'] 
    #   for p in parameters:
    #        inpt_df.loc[inpt_df[p] < -9990.,p] = np.nan
    #        inpt_df[p].ffill(inplace=True)
    #

    #Do parameter calculation of different cadences
    plsm_df['del_time_pls'] = np.abs(plsm_df['time_dt_pls'].diff(1).values.astype('double')/1.e9)
    magf_df['del_time_mag'] = np.abs(magf_df['time_dt_mag'].diff(1).values.astype('double')/1.e9)


    #convert span to a number index so I can use the logic center = True
    #assumes format_df import is in s
    #then divide by the space craft jump time
    if center:
        span_mag = int(round(float(span[:-1])/magf_df.del_time_mag.median()))
        span_pls = int(round(float(span[:-1])/plsm_df.del_time_pls.median()))
        print('######################################################')
        print(span,span_mag,span_pls,plsm_df.del_time_pls.median(),magf_df.del_time_mag.median())
        print('######################################################')
    #else use the runup (left) span
    else:
        span_mag = span
        span_pls = span

    #time cadence parameter to add to plasma and magnetic field time series
    par_ind_mag = magf_df.del_time_mag.median()**2./60./magf_df.del_time_mag
    par_ind_pls = plsm_df.del_time_pls.median()**2./60./plsm_df.del_time_pls

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
    magf_df.loc[:,m_var] = logit_model(magf_df,-10.0575,0.6861,0.7071,0.6640,plasma=False)

    #cur arrays to the bar minimum for matching
    plsm_df = plsm_df.loc[:,[p_var,'diff_med_speed', 'diff_med_Np', 'diff_med_Vth','diff_sig_speed', 'diff_sig_Np', 'diff_sig_Vth']]
    magf_df = magf_df.loc[:,[m_var,'diff_med_Bx', 'diff_med_By', 'diff_med_Bz', 'diff_sig_Bx', 'diff_sig_By', 'diff_sig_Bz']]
    
    #update array with new p-values by matching on indices
    outp_df  = pd.merge(inpt_df,plsm_df,how='left',left_index=True,right_index=True,sort=True)
    outp_df  = pd.merge(outp_df,magf_df,how='left',left_index=True,right_index=True,sort=True)


    #forward fill NaN event probabilities
    #outp_df[p_var].ffill(inplace=True)
    #outp_df[m_var].ffill(inplace=True)

    return outp_df


#probability model and parameters to use
def logit_model(df,b0,b1,b2,b3,plasma=True):
    '''
    Returns a probability value for a set of input parameters

    Parameters
    ----------
    df: Pandas DataFrame
        Pandas DataFrame containing plasma or magnetic field paraemters
    b0: float
        Logit model intercept
    b1: float
        Logit model slope for parameter 1 (either Bx or proton flow speed) 
    b2: float
        Logit model slope for parameter 2 (either By or proton thermal velocity)
    b3: float
        Logit model slope for parameter 3 (either Bz or proton number density)
    plasma: boolean, optinal
        Use plasma parametes for model (Default = True)  or 
        use magnetic field parameters (plasma=False)
    
    Returns
    -------
    logit probability for a given set of parameters

    '''
    if plasma:
        return 1./(np.exp(-(df.diff_snr_speed*b1+df.diff_snr_Vth*b2+df.diff_snr_Np*b3+b0))+1.) 
    else:
        return 1./(np.exp(-(df.diff_snr_Bx*b1+df.diff_snr_By*b2+df.diff_snr_Bz*b3+b0))+1.) 

def help_chi_min(args):
    '''
    Allows for multiple arguments to be called by a Pooled Multiprocessor function
    
    Parameters
    -----------
    args: list
        A list of parameters in order of return_chi_min

    Return
    -----------
    time,chisq : Tuple
        Result of return_chi_min
    
    
    '''

    try:
        return return_chi_min(*args)
    except:
        return args[7],np.nan


def dumb_chi_min(args):
    print(args[-1],current_process())
    #rgh_chi_t = args[0]
    #plsm      = args[1]
    #k         = args[2]
    #par       = args[3]
    #try_mag   = args[4]
    #try_pls   = args[5]
    #trainer_time = args[6]
    #time      = args[7]
    #trainer='Wind'
    chisq = np.random.random(1)

    return args,chisq

#parallize chi^2 computation
def return_chi_min(rgh_chi_t,plsm,k,par,try_mag,try_pls,trainer_time,time,trainer='Wind'):

    """
    return_chi_min computes the chi^2 min. time for a given set of parameters
    Parameters
    -----------
    rgh_chi_t: Pandas datetime delta object
        Time around a given time in p_mat to include in chi^2 minimization  
    plsm: dict
        Dictionary of plasma DataFrames of plasma Dataframes
    k   : string
        Spacecraft name in plsm dictionary
    par : string,list
        List of string of parameters to initially try and compute Chi^2 min on
    try_mag: boolean
        If true try to force using magnetic field for matching
    try_pls: boolean
        If true try to force flow speed matching 
    trainer_time: Time index
        Time index of training spacecraft to match and get Chi^2 min.
    time: Time index
        Time index of nonprimary spacecraft to offset and match with primary spacecraft
    trainer: string
        The spacecraft to use for X^2 comparison (Default=Wind)

    RETURNS
    --------
    time,chisq : datetime index, calculate Chi^2
    Datetime index of  Chi^2 time and the Chi^2 value

    """
   

    #get a region around one of the best fit times
    com_slice = [time-rgh_chi_t,time+rgh_chi_t]
    c_mat = plsm[k].loc[com_slice[0]:com_slice[1]]

    #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
    c_mat.index = c_mat.index+(trainer_time-time)


    #c_mat.SPEED.interpolate('time',inplace=True)
    #need to use values for correct logic

    #get training spacecraft time range
    t_mat = plsm[trainer].loc[c_mat.index.min():c_mat.index.max()]

    #use magentic field for refinement for ACE, Wind, and DSCOVR
    if ((k.lower() != 'soho') & (try_mag)): par = ['Bx','By','Bz']

    #use speed for rough esimation if possible
    if (((len(c_mat[c_mat['SPEED'] > 0].SPEED) > 10.) & (len(t_mat[t_mat['SPEED'] > 0].SPEED) > 10.) & (try_pls)) | (k.lower() == 'soho')): par = ['SPEED']
    else: par = ['Bx','By','Bz']


    #remove bad Speed values 
    c_mat.loc[c_mat.SPEED < 0.,'SPEED'] = np.nan


    #need to use finite values for correct logic
    c_mat = c_mat[np.isfinite(c_mat[par].values)]
    t_mat = t_mat[np.isfinite(t_mat[par].values)]

    #drop duplicates when using more than one array value
    #use mag column because in has the higher sampling fequence
    c_mat = c_mat[~c_mat.index.duplicated(keep='first')]
    t_mat = t_mat[~t_mat.index.duplicated(keep='first')]


    #go to next time if DataFrame is empty
    if ((len(c_mat) < 2) | (len(t_mat) < 2)): 
        return time,np.nan


    #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid and interpolate
    c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')


    #get the median slope and offset
    #J. Prchlik (2017/11/20)
    try:
        med_m,med_i,low_s,hgh_s = theilslopes(t_mat.SPEED.values,c_mat.SPEED.values)
        c_mat.SPEED = c_mat.SPEED*med_m+med_i
    except IndexError:
    #get median offset to apply to match spacecraft
        off_speed = c_mat.SPEED.median()-t_mat.SPEED.median()
        c_mat.SPEED = c_mat.SPEED-off_speed
    

    #sometimes different componets give better chi^2 values therefore reject the worst when more than 1 parameter
    #Try using the parameter with the largest difference  in B values preceding and including the event (2017/12/11 J. Prchlik)
    if len(par) > 1:
       #par_chi = np.array([np.sum((((c_mat.loc[:,par_i]-t_mat.loc[:,par_i])/t_mat.loc[:,par_i].median())**2.).values)/float(len(c_mat)+len(t_mat)) for par_i in par])
       #use_par, = np.where(par_chi == np.min(par_chi))
       par_chi = np.array([(t_mat.loc['1950/10/10':trainer_time,par_i].diff().abs().max()) for par_i in par])
       use_par, = np.where(par_chi == np.max(par_chi))
       par      = list(np.array(par)[use_par])
       #par = ['Bt']


    #compute locate variation in parameters
    #from 15 pixels around the observation [J. Prchlik 2017/12/11]
    loc_var = 15
    t_mat['lc_std_'+par[0]] = t_mat[par].rolling(loc_var,center=True ,min_periods=1).std()**2.
    c_mat['lc_std_'+par[0]] = c_mat[par].rolling(loc_var,center=True ,min_periods=1).std()**2.


    #fill the first values
    t_mat['lc_std_'+par[0]].fillna(method='bfill',inplace=True)
    c_mat['lc_std_'+par[0]].fillna(method='bfill',inplace=True)
    t_mat['lc_std_'+par[0]].fillna(method='ffill',inplace=True)
    c_mat['lc_std_'+par[0]].fillna(method='ffill',inplace=True)

    #compute chi^2 value for Wind and other spacecraft
    #added computation number to prefer maximum overlap (i.e. won't shove to an edge) J. Prchlik 2017/11/15
    #Added uncertainty based on locale variables 2017/12/11 (J. Prchlik)
    #add logic to only get run upto event if using mag. fields 2017/10/11 J. Prchlik
    if par[0] == 'SPEED':
        chisq = np.sum(((c_mat.loc[:,par]-t_mat.loc[:,par])**2.).values/(t_mat['lc_std_'+par[0]].values+c_mat['lc_std_'+par[0]].values))
    else:
        pad = pd.to_timedelta('1 minutes')
        chisq = np.sum(((c_mat.loc[:trainer_time+pad,par]-t_mat.loc[:trainer_time+pad,par])**2.).values/(t_mat['lc_std_'+par[0]].values+c_mat['lc_std_'+par[0]].values))
    #Try using the median offset rather than chi^2 minimum (Did not work)
    #chisq = np.nanmedian(((c_mat.loc[:,par]-t_mat.loc[:,par])**2.).values)/float(len(c_mat)+len(t_mat))
     

    return time,chisq


#function to find the Chi^2 min value given a set of parameters
def chi_min(p_mat,par,rgh_chi_t,plsm,k,window,ref_window,trainer_t,color,marker,ref_chi_t=pd.to_timedelta('10 minutes'),refine=True,n_fine=4,plot=True ,nproc=1,use_discon=False):
    """
    chi_min computes the chi^2 min. time for a given set of parameters
    Parameters
    -----------
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
    window : dictionary
        Spacecraft windows to use when matching
    ref_window : dictionary
        Spacecraft windows to use when refined matching
    trainer_t: Time index
        Time index of training spacecraft to match and get Chi^2 min.
    color: dictionary
        Dictionary of spacecraft colors (keys should be k)
    marker: dictionary
        Dictionary of spacecraft markers (keys should be k)
    refine: boolean
        Whether to use a refined window (Default = True)
    n_fine : integer
        Number of chi^2 min values from the rough grid to check with the fine grid (Default = 4)
    plot  : boolean,optional
        Plot Chi^2 minimization window (Default = False)
    use_discon : boolean,optional
        Only use discontinuity (Default = False)

    RETURNS
    --------
    i_min : datetime index
    Datetime index of best fit Chi^2 time

    """


    #if plot create plotting axis
    if plot:
        chi_fig,chi_ax = plt.subplots()

    #set up chisquared array in pandas object
    p_mat.loc[:,'chisq'] = np.nan 
    #keep one for plotting range
    p_mat_r = p_mat.copy()

    #first set to no time offset
    i_min = trainer_t


    #loop and squeeze rough window
    for j in range(2):

        #Dont loop if use_discon 
        if ((j == 1) & (use_discon)): continue
        #looping time around window to try to match
        t_rgh_wid = window[k]/(j+1)
        #looping range of window size
        t_rgh_chi_t = ref_chi_t/(j+1)

        #get all values in top n_fine at full resolution
        p_mat  = plsm[k].loc[i_min-t_rgh_wid:i_min+t_rgh_wid]
        #Use Flow Speed Cadence for rough esitmation
        pt_mat  = p_mat[np.isfinite(p_mat.SPEED)]
        #if no SPEED Values then use Total magentic field downsampled by 5
        if len(p_mat) < 5: pt_mat = p_mat[np.isfinite(p_mat.Bt)].iloc[::5,:]

        #list of to values to compute X^2 minium
        time = p_mat.index
   

        #Just use discontinuity
        if use_discon: time=p_mat_r.index
        

        #create list to sent to processors
        #par_chi_min = partial(return_chi_min,rgh_chi_t,plsm,k,par,False,True,trainer_t)
   
        #Commented out by J. Prchlik (2017/11/29 for threading testing)
        loop_list = []
        for l in time: loop_list.append((t_rgh_chi_t,plsm,k,par,False,True,trainer_t,l))
        #parallel issue is with using a pandas Dataframe (need to work out solution)
        #actually just has issue with really large arrays. No clear solution  other 
        #than be annoyed for now
        #for l in time: loop_list.append((t_rgh_chi_t,plsm,k,par,False,True,trainer_t,l))
        #loop_list = []
        #for l in time: loop_list.append((0))
        

        #clear Threads
        #print('Number of Threads')
        #print(threading.active_count())
        #all_th = threading.enumerate()
        #for i in all_th: i.clear()



        if nproc > 1.5:
            pool2 = Pool(processes=nproc)
            #Commented out by J. Prchlik (2017/11/29 for threading testing)
            #outp = pool2.map(dumb_chi_min,loop_list)
            outp = pool2.map(help_chi_min,loop_list)
            pool2.close()
            pool2.join()
        else:
            outp = []
            for l in loop_list: outp.append(help_chi_min(l))


        #add chisq times to p_mat array
        #first is time index second is chisq value
        for l in outp: p_mat.loc[l[0],'chisq'] = l[1]

        
        
        #get the index of minimum chisq value
        i_min = p_mat['chisq'].idxmin()

        #plot chi^2 min
        if plot: chi_ax.scatter(p_mat.index,p_mat.chisq/p_mat.chisq.min(),label=k,color=color[k],marker=marker[k])
    
    #use fine grid around observations to match time offset locally
    if refine:

        #use magentic field for refinement for ACE, Wind, and DSCOVR
        if k.lower() != 'soho': par = ['Bx','By','Bz']
        else: par = ['SPEED']

        #Find 4 lowest chisq min times 
        #p_temp = p_mat.sort_values('chisq',ascending=True )[:n_fine]

        #set the min value to interative solve for refinmenet
        #i_min = p_temp['chisq'].idxmin()
 
        #loop and squeeze refinement window
        for j in range(3):
            #interatively solve around 1 sigma
            #2017/12/11 J. Prchlik
            i_sig = p_mat.loc[p_mat['chisq'] < 2.*p_mat['chisq'].min(),:]

            #looping time around window to try to match
            t_ref_wid = ref_window[k]/(j+1)
            #looping range of window size
            t_ref_chi_t = ref_chi_t/(j+1)

            #get all values in top n_fine at full resolution
            #p_mat  = plsm[k].loc[i_min-t_ref_wid:i_min+t_ref_wid]
            #print(p_temp.index.min()-rgh_chi_t,p_temp.index.max()+rgh_chi_t)
            #interatively solve around 1 sigma
            #2017/12/11 J. Prchlik
            p_mat  = plsm[k].loc[i_sig.index.min():i_sig.index.max()]
 
            #list of to values to compute X^2 minium
            time = p_mat.index
            #create list of tuples to sent to processors
            #create list to sent to processors
            #ref_chi_min = partial(return_chi_min,ref_chi_t,plsm,k,par,True,False,trainer_t)
            loop_list = []
            #try only using SPEED 2017/12/11
            for i in time: loop_list.append((t_ref_chi_t,plsm,k,par,True ,False,trainer_t,i))
    
            #Parallized chisq computation
            if nproc > 1.5:
                pool3 = Pool(processes=nproc)
                outp = pool3.map(help_chi_min,loop_list)
                pool3.terminate()
                pool3.close()
                pool3.join()
            else:
                outp = []
                for i in loop_list: outp.append(help_chi_min(i))


            #add chisq times to p_mat array
            #first is time index second is chisq value
            for i in outp: p_mat.loc[i[0],'chisq'] = i[1]
            #get the index of minimum refined chisq value
            i_min = p_mat['chisq'].idxmin()

      
            #variable to exit while loop
            looper = 1

            #check to see if chisq min is at an edge but if it is still at an edge after 3 tries give up
            while (((i_min == p_mat.index.max()) | (i_min == p_mat.index.min())) & (looper < 4)):
                #get all values in top n_fine at full resolution
                p_mat  = plsm[k].loc[p_mat.index.min()-(t_ref_wid*(looper+1)):p_mat.index.max()+(t_ref_wid*(looper+1))]
                #list of to values to compute X^2 minium
                time = p_mat.index
                loop_list = []
                #create list to pass to function
                for i in time: loop_list.append((t_ref_chi_t,plsm,k,par,True,False,trainer_t,i))

                #Parallized chisq computation
                if nproc > 1.5:
                    pool4 = Pool(processes=nproc)
                    outp = pool4.map(help_chi_min,loop_list)
                    pool4.terminate()
                    pool4.close()
                    pool4.join()
                else:
                    outp = []
                    for i in loop_list: outp.append(help_chi_min(i))


                #add chisq times to p_mat array
                #first is time index second is chisq value
                for i in outp: p_mat.loc[i[0],'chisq'] = i[1]
                #get the index of minimum refined chisq value
                i_min = p_mat['chisq'].idxmin()
  
                #Add 1 to running loop
                looper += 1

            #get the index of minimum refined chisq value
            i_min = p_mat['chisq'].idxmin()
            #plot chi^2 min
            if plot: chi_ax.scatter(p_mat.index,p_mat.chisq/p_mat.chisq.min(),label='Ref. {0:1d} {1} '.format(j+1,k),marker=marker[k])

    
    
    #set up plot for chi^2 min
    if plot:
        chi_ax.legend(loc='best',frameon=False,scatterpoints=1)
        chi_ax.set_xlim([p_mat_r.index.min()-pd.to_timedelta('30 minutes'),p_mat_r.index.max()+pd.to_timedelta('30 minutes')])
        chi_ax.set_ylabel('$\chi^2$')
        chi_ax.set_xlabel('Time [UTC]')
        #set plot maximum to be 10% higher than the max value
        #ymax = 1.1*np.nanmax(p_mat.chisq.values/p_mat.chisq.min())
    
        ##if ymax > 10 then set ymax to 10
        #if ymax > 10.: ymax = 10.
        ymax= 10
        
        chi_ax.set_ylim([0.5,ymax])
        fancy_plot(chi_ax)
        chi_fig.savefig('../plots/spacecraft_events/chisq/chi_min_{0:%Y%m%d_%H%M%S}_{1}.png'.format(trainer_t,k.lower()),bbox_pad=.1,bbox_inches='tight')
        #ax.set_ylim([300,1000.])
        plt.close(chi_fig)
   
    #get upper and lower bounds in Chi^2
    i_upp = (p_mat.loc[i_min:'2100/10/01','chisq']-2.*p_mat.loc[i_min,'chisq']).abs().idxmin()
    i_low = (p_mat.loc['1900/10/01':i_min,'chisq']-2.*p_mat.loc[i_min,'chisq']).abs().idxmin()

    #return best fit, 1sigma upper limit, 1 sigma lower limit
    return i_min,i_upp,i_low
    
def main(craft=['Wind','DSCOVR','ACE','SOHO'],col=['blue','black','red','teal'],mar=['D','o','s','<'],
         use_craft=False,use_chisq=True,use_discon=False,plot=True,refine=True ,verbose=True,nproc=1,p_val=0.999,
         ref_chi_t = pd.to_timedelta('30 minutes'),rgh_chi_t = pd.to_timedelta('90 minutes'),
         arch='../cdf/cdftotxt/',mag_fmt='{0}_mag_2015_2017_formatted.txt',pls_fmt='{0}_pls_2015_2017_formatted.txt',
         start_t='2017/11/17',end_t='2017/07/31',center=False):
    '''
    Function which finds solar wind events in a specific spacecraft. Then the program does a Chi^2 minimization to find the event
    in the other spacecraft. In doing so it creates a series of plots and a html table to summarize the results.

    Parameters
    -----------
    craft: list, optional
        List of spacecraft to analyze for events. The first spacecraft in the list finds the events and is the reference point
        for the Chi^2 minimization (Default = ['Wind','DSCOVR','ACE','SOHO']).
    col  : list, optional
        List of matplotlib colors for plotting spacecraft points. The color list index corresponds to the index in craft
        (Default = ['blue','black','red','teal']).
    mar  : list, optional
        List of matplotlib markers for plotting spacecraft points. The marker list index corresponds to the index in craft
        (Default = ['D','o','s','<']).
    use_craft : boolean, optional
        Use spacecraft positions to restrict finding range (Default = False). It turns out a lot of events are not radial 
        therefore use_craft is no useful. 
    use_chisq : boolean, optional
        Use Chi^2 minimization to find corresponding events between spacecraft. The Chi^2 minimum value relates to a given 
        spacecraft and the first spacecraft in the craft list (Default = True).
    use_discon: boolean, optional
        Use Chi^2 minimization to find corresponding events between spacecraft, but only set on discontinuities. 
        The Chi^2 minimum value relates to a given spacecraft and the first spacecraft in the craft list (Default = False).
    plot      : boolean, optional
        Plot the Chi^2 minimum values as a function of time for a given spacecraft with respect to the reference spacecraft
        (Default = True). You should keep this parameter on because the output html file references this created file.
    refine:  boolean, optional
        Refine Chi^2 minimization by loop with smaller ranges,windows around previous minimum at the highest possible
        observational cadence (Default = True).
    verbose: boolean, optional
        Print the process time for an event (Default = True).
    nproc  : integer, optional
        Number of processors to use in Chi^2 minimization (Default = 1). For 1 event I see a 30% time decrease by using
        nproc=8 over nproc=1.
    p_val : float, optional
        Probability value (0<p_val<1.0) used to define an event (Default = 0.999). Do not set value below 0.90 for two reasons.
        One that creates a lot of "bad events" (i.e. events not seen in all four spacecraft or use instrument spikes.) two 
        currently the code has a cut of 0.90 to define an event start.
    ref_chi_t : pandas time delta object, optional
        Size of window around an event (+/-) to find the Chi^2 minimum for a refined window (Default = pd.to_timedelta('30 minutes')).
        During refinement the window will run once and shrink two times by 1+loop (Default = 30,15,10).
    rgh_chi_t : pandas time delta object, optional
        Size of window around an event (+/-) to find the Chi^2 minimum for a rough window (Default = pd.to_timedelta('90 minutes')).
        During the rough minimization the window will run once and shrink once by 1+loop (Default = 90,45).
    arch: string, optional
        The archive location for the text file to read in (Default = '../cdf/cdftotxt/')
    mag_fmt: string, optional
        The file format for the magnetic field observations (Default = '{0}_mag_formatted.txt',
        where 0 is the formatted k).
    pls_fmt: string, optional
        The file format for the plasma observations (Default = '{0}_pls_formatted.txt',
        where 0 is the formatted k).
    start_t: string, optional
        Date in YYYY/MM/DD format to start looking for events (Default = '2016/06/04')
    start_t: string, optional
        Date in YYYY/MM/DD format to stop looking for events (inclusive, Default = '2017/07/31')
    center = boolean, optional
        Whether the analyzed point to be center focused (center = True) or right focus (Default = False).
        Using a right focused model I find better agreement with events found by eye.

    Returns
    -------
    '''
    #dictionary for storing the Pandas Data frame
    pls = {}
    mag = {}
    plsm = {}
    
    
    #use space craft position to get range of time values
    #(use_craft)
    
    
    #use best chisq to find the best time in between spacecraft
    #(use_chisq)
    
    #plot chisq min procedure
    #(plot)
    
    #refine chisq min with closer time grid
    #(refine)
    
    #set use to use all spacecraft
    
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
    m_var = p_var.replace('predict','predict_sigma')
    #fractional p value to call an "event"
    #(p_val) = 0.950 
    
    #get strings for times around each event#
    window = {}
    window['DSCOVR'] = pd.to_timedelta('40 minutes')
    window['ACE'] = pd.to_timedelta('40 minutes')
    window['SOHO'] = pd.to_timedelta('40 minutes')
    window['Wind'] = pd.to_timedelta('40 minutes')

    if use_discon:
        #get strings for times around each event if using donconitinity matching#
        window = {}
        window['DSCOVR'] = pd.to_timedelta('90 minutes')
        window['ACE'] = pd.to_timedelta('90 minutes')
        window['SOHO'] = pd.to_timedelta('90 minutes')
        window['Wind'] = pd.to_timedelta('90 minutes')
        
    
    #define rough chi min time  to cal Chi^2 min for each time
    #(rgh_chi_t = pd.to_timedelta('90 minutes')
    
    #get strings for times around each event when refining chi^2 time
    ref_window = {}
    #ref_window['DSCOVR'] = pd.to_timedelta('15 minutes')
    #ref_window['ACE'] = pd.to_timedelta('15 minutes')
    #ref_window['SOHO'] = pd.to_timedelta('25 minutes')
    #ref_window['Wind'] = pd.to_timedelta('25 minutes')
    ref_window['DSCOVR'] = pd.to_timedelta('5 minutes')
    ref_window['ACE'] = pd.to_timedelta('5 minutes')
    ref_window['SOHO'] = pd.to_timedelta('5 minutes')
    ref_window['Wind'] = pd.to_timedelta('5 minutes')
    
    #refined window to calculate Chi^2 min for each time
    #(ref_chi_t)
    #Parameters for file read in and parsing
    par_read_in = partial(read_in,arch=arch,mag_fmt=mag_fmt,pls_fmt=pls_fmt,start_t=start_t,end_t=end_t,center=center)
                         


    #plot window 
    plt_windw = pd.to_timedelta('180 minutes')
    
    #window around event to get largers parameter jump values
    # when writing to file
    a_w = pd.to_timedelta('100 seconds')
    
    #read in and format spacecraft in parallel
    pool = Pool(processes=4)
    outp = pool.map(par_read_in,craft)
    pool.terminate()
    pool.close()
    pool.join()
    
    
    #create global plasma key
    for i in outp:
        plsm[i.craft.values[0]] = i
    
    
    #use the trainer space craft to find events
    tr_events = plsm[trainer][plsm[trainer][p_var] > p_val]
    
    #Group events by 1 per hour
    #ctr_events['str_hour'] = tr_events.time_dt_mag.dt.strftime('%Y%m%d%H')
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
    
    #For an hour around an event find the minimum time to get a precise break point using a 10% lower prob. threshold
    #search window for the star
    s_wind = pd.to_timedelta('30 minutes')
    for i in tr_events.index:
          temp_events = plsm[trainer][i-s_wind:i+s_wind][plsm[trainer][i-s_wind:i+s_wind][p_var] >.90]
          tr_events.loc[i,:] = temp_events.loc[temp_events.index.min(),:]
    
    #reset index with new plasma time value
    tr_events.set_index(tr_events.time_dt_pls,inplace=True)
    
    
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
                discontinuity in a component of the magnetic field. Then I create a course grid of solar wind speed samples for &plusmn; 40 minutes around a event.
                Currently, I find {1:1d} events using the Wind spacecraft to define an event. Then I shift 
                the other 3 spacecraft onto the Wind using the course grid. Then I select a 80 minute (&plusmn;40 minutes) window around the shifted Wind and other spacecraft and derive a first order
                &chi;<sup>2</sup> minimum. Using the first order &chi;<sup>2</sup> minimum, I create a fine grid 50 minutes (&plusmn; 25 minutes) around the rough &chi;<sup>2</sup> minimum
                time. The fine grid cadence is set to the maximum sampling frequency of a given spacecraft. Then the program stores the &chi;<sup>2</sup> value for 20 minutes (&plusmn; 10 minutes)
                around each fine grid point. However, if the first or last point is the &chi;<sup>2</sup> min. then the program expands the time search window by 20 minutes.
                The &chi;<sup>2</sup> reported is the minimum from the time grid.
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
                while an empty cell denotes using the magnetic field. Clicking on the spacecraft shows the &chi;<sup>2</sup> minimization as a function of time for 
                the wide and refined windows.
        
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
                      Pos. Unc. [s]
                      </td>
                      <td>
                      Neg. Unc. [s]
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
                      <td>
                      &delta;SPEED [km/s]
                      </td>
                      <td>
                      &delta;Np [cm<sup>-3</sup>]
                      </td>
                      <td>
                      &delta;Vth [km/s]
                      </td>
                      <td>
                      &delta;Bx [nT]
                      </td>
                      <td>
                      &delta;By [nT]
                      </td>
                      <td>
                      &delta;Bz [nT]
                      </td>
                  </tr>
                      '''
    
    #data format for new row
    new_row =   '''<tr>
                      <td>
                      <a href="../plots/spacecraft_events/chisq/chi_min_{8:%Y%m%d_%H%M%S}_{9}.png">{0}</a>
                      </td>
                      <td>
                      {1:%Y/%m/%d %H:%M:%S}
                      </td>
                      <td>
                      {2:5.2f}
                      </td>
                      <td>
                      {3:5.2f}
                      </td>
                      <td>
                      {4:5.2f}
                      </td>
                      <td>
                      {5:4.3f}
                      </td>
                      <td>
                      {6:4.3f}
                      </td>
                      <td>
                      {7}
                      </td>
                      <td>
                      {10:4.3f}
                      </td>
                      <td>
                      {11:4.3f}
                      </td>
                      <td>
                      {12:4.3f}
                      </td>
                      <td>
                      {13:4.3f}
                      </td>
                      <td>
                      {14:4.3f}
                      </td>
                      <td>
                      {15:4.3f}
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
    
    
    #Additional output parameters
    par_out = ['diff_med_speed', 'diff_med_Np', 'diff_med_Vth', 'diff_med_Bx', 'diff_med_By', 'diff_med_Bz']
    
    
    # write header for html page
    out_f = open('../html_files/{0}_level_{1:4.0f}_full_res.html'.format(trainer.lower(),p_val*1000.).replace(' ','0'),'w')
    out_f.write(ful_hdr+par_hdr.format(p_val*100.,len(tr_events)))
    
    
    #get event slices 
    for i in tr_events.index:
        #print processing time
        start_time = time.time()
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
        out_f.write(new_row.format(trainer,i,0.00,0.00,0.00,tr_events.loc[i,p_var],tr_events.loc[i-a_w:i+a_w,p_var.replace('predict','predict_sigma')].max(),'X',i,trainer.lower(),*plsm[trainer].loc[i-a_w:i+a_w,par_out].max()))
    
    
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
                   #print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max={1:4.3f}'.format((i_max-i).total_seconds()/60.,p_mat.loc[i_max][p_var],i_max))
                else:
                   print('No Plasma Observations')
      
            #elif use_chisq:
         
            #use the largets set of discontinuties 
            elif ((use_discon) | (use_chisq)):
                #parameters if using discontinuity
                if use_discon:
                     #calculate difference in parameters
                     p_mat['del_speed'] = np.abs(p_mat['SPEED'].diff(1))
                     p_mat['del_Np'] = np.abs(p_mat['Np'].diff(1))
                     p_mat['del_Vth'] = np.abs(p_mat['Vth'].diff(1))
                     p_mat['del_Bx'] = np.abs(p_mat['Bx'].diff(1))
                     p_mat['del_By'] = np.abs(p_mat['By'].diff(1))
                     p_mat['del_Bz'] = np.abs(p_mat['Bz'].diff(1))

                     #find largest magnetic field discontinuties 
                     p_mat['del_Bt'] = np.sqrt(p_mat.del_Bx**2.+p_mat.del_By**2.+p_mat.del_Bz**2.) 

                     #find largest speed discontinuties 
                     p_mat_t = p_mat.sort_values('del_speed',ascending=False)[:10]

             
                     #find largest mag field discontinuies 
                     p_mag_t = p_mat.sort_values('del_Bt',ascending=False)[:10]
    
                     #mag tolerance for using magnetometer data to match events rather than plasma parameters
                     mag_tol = 0.0

                #use chisq minimum of top events in 2 hour window
                elif use_chisq:
                    #downsample to 5 minutes for time matching in chisq
                    #remove downsampling J. Prchlik 2017/11/20
                    p_mat_t = p_mat #.resample(downsamp).median()
                    #p_mat_t = p_mat.sort_values(p_var,ascending=False)[:40]
                    #p_mat_t = p_mat[p_var].idxmax()
    
                    #downsample mag to 5 minutes for time matching in chisq
                    #p_mag_t = p_mat.sort_values(m_var,ascending=False)[:40]
                    #p_mat_t = p_mat[p_var].idxmax()
                    p_mag_t = p_mat_t
    
                    #mag tolerance for using magnetometer data to match events rather than plasma parameters
                    mag_tol = 0.0
               

                

                #temporary plasma array to use for X^2 matching
                t_plsm = {}
                half_day = pd.to_timedelta('9 hours')
                for ll in plsm.keys(): t_plsm[ll] = plsm[ll].loc[i-half_day:i+half_day]
    
    
                if (((p_mat_t.size > 0) & (p_mat_t[p_var].max() > mag_tol)) | ((k.lower() == 'soho') & (p_mat_t.size > 0.))):
                #if ((k.lower() == 'soho') & (p_mat_t.size > 0.)):
    
                    i_min,i_upp,i_low = chi_min(p_mat_t,['SPEED'],rgh_chi_t,t_plsm,k,window,ref_window,i,color,marker,ref_chi_t=ref_chi_t,
                                    refine=refine,n_fine=1,plot=plot,nproc=nproc,use_discon=use_discon)
                    #use full array for index matching
                    p_mat = plsm[k]
    
                    try:
     
                        if k.lower() == 'soho':
                            #print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min))
                            out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,(i_upp-i_min).total_seconds(),(i_low-i_min).total_seconds(),p_mat.loc[i_min-a_w:i_min+a_w][p_var].max(),0.000,'X',i,k.lower(),*p_mat.loc[i_min-a_w:i_min+a_w,par_out].max()))
    
                        else: 
                            #print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
                            out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,(i_upp-i_min).total_seconds(),(i_low-i_min).total_seconds(),p_mat.loc[i_min-a_w:i_min+a_w][p_var].max(),p_mat.loc[i_min-a_w:i_min+a_w][p_var.replace('predict','predict_sigma')].max(),'X',i,k.lower(),*p_mat.loc[i_min-a_w:i_min+a_w,par_out].max()))
    
                    except KeyError:
                        print('Missing Index')
    
    
                #else no plasma observations                                                                                                                                                      
                elif (((p_mat_t.size == 0) | (p_mat_t[p_var].max() <= mag_tol) | (np.isnan(p_mat_t[p_var].max()))) & (p_mag_t.size > 0.) & (k.lower() != 'soho')):
                #elif ((p_mag_t.size > 0.) & (k.lower() != 'soho')):
                    #sort the cut window and get the top 10 events
                    p_mat_t = p_mag_t
           
                    i_min,i_upp,i_low = chi_min(p_mat_t,['Bx','By','Bz'],rgh_chi_t,t_plsm,k,window,ref_window,i,color,marker,
                                    ref_chi_t=ref_chi_t,refine=refine,n_fine=1,plot=plot,nproc=nproc,use_discon=use_discon)
    
                    #use full array for index matching
                    p_mat = plsm[k]
                    try:
                    #print output to terminal
                       # print('{2:%Y/%m/%d %H:%M:%S},{0:5.2f} min., p_max (plsm) ={1:4.3f}, p_max (mag) = {3:4.3f}'.format((i_min-i).total_seconds()/60.,p_mat.loc[i_min][p_var],i_min,p_mat.loc[i_min][p_var.replace('predict','predict_sigma')]))
                        out_f.write(new_row.format(k,i_min,(i_min-i).total_seconds()/60.,(i_upp-i_min).total_seconds(),(i_low-i_min).total_seconds(),p_mat.loc[i_min-a_w:i_min+a_w][p_var].max(),p_mat.loc[i_min-a_w:i_min+a_w][p_var.replace('predict','predict_sigma')].max(),'',i,k.lower(),*p_mat.loc[i_min-a_w:i_min+a_w,par_out].max()))
    
    
                    except KeyError:
                        print('Missing Index')
           
                else:
                   print('No Plasma or Mag. Observations')
                   continue


            #get a region around one of the best fit times
            plt_slice = [i_min-plt_windw,i_min+plt_windw]
            b_mat = t_plsm[k].loc[plt_slice[0]:plt_slice[1]]
            
            #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
            b_mat.index = b_mat.index+(i-i_min)
    
            #plot plasma parameters

            #get a region around one of the best fit times
            plt_slice = [i_min-plt_windw,i_min+plt_windw]
            b_mat = t_plsm[k].loc[plt_slice[0]:plt_slice[1]]
            
            #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
            b_mat.index = b_mat.index+(i-i_min)
    
            #plot plasma parameters
            #fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k.upper())         
            if len(b_mat[b_mat['Np'   ] > -9990.0]) > 0:
                fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k)         
                fax[0,0].plot(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,color=color[k],linewidth=2,label='')         
    
            if len(b_mat[b_mat['Vth'  ] > -9990.0]) > 0:
                fax[1,0].scatter(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[k],color=color[k],label=k)         
                fax[1,0].plot(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,color=color[k],linewidth=2,label='')         
    
            if len(b_mat[b_mat['SPEED'] > -9990.0]) > 0:
                fax[2,0].scatter(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,marker=marker[k],color=color[k])         
                fax[2,0].plot(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,color=color[k],linewidth=2)         
    
    
            #plot mag. parameters
            if k.lower() != 'soho':
                if len(b_mat[b_mat['Bx']    > -9990.0]) > 0:
                    fax[0,1].scatter(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,marker=marker[k],color=color[k])         
                    fax[0,1].plot(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,color=color[k],linewidth=2)         
    
                if len(b_mat[b_mat['By']    > -9990.0]) > 0:
                    fax[1,1].scatter(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,marker=marker[k],color=color[k])         
                    fax[1,1].plot(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,color=color[k],linewidth=2)         
    
                if len(b_mat[b_mat['Bz']    > -9990.0]) > 0:
                    fax[2,1].scatter(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,marker=marker[k],color=color[k])         
                    fax[2,1].plot(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,color=color[k],linewidth=2)         
    
            #print separater 
            print('########################################')
        
        #print end comupation time per event
        if verbose: print("--- Process time for Event %s seconds ---" % (time.time() - start_time))
      
        #get training spacecraft time range
        plt_slice = [i-plt_windw,i+plt_windw]
        t_mat = t_plsm[trainer].loc[plt_slice[0]:plt_slice[1]]
    
    
    
        #plot plasma parameters
        if len(t_mat[t_mat['Np'   ] > -9990.0]) > 0:
            fax[0,0].scatter(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,marker=marker[trainer],color=color[trainer],label=trainer.upper())         
            fax[0,0].plot(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,color=color[trainer],linewidth=2,label='')         
    
        if len(t_mat[t_mat['Vth'  ] > -9990.0]) > 0:
            fax[1,0].scatter(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[trainer],color=color[trainer],label=trainer)         
            fax[1,0].plot(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,color=color[trainer],linewidth=2,label='')         
    
        if len(t_mat[t_mat['SPEED'] > -9990.0]) > 0:
            fax[2,0].scatter(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,marker=marker[trainer],color=color[trainer])         
            fax[2,0].plot(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,color=color[trainer],linewidth=2)         
    
    
        #plot mag. parameters
        if len(t_mat[t_mat['Bx']    > -9990.0]) > 0:
            fax[0,1].scatter(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,marker=marker[trainer],color=color[trainer])         
            fax[0,1].plot(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,color=color[trainer],linewidth=2)         
    
        if len(t_mat[t_mat['By']    > -9990.0]) > 0:
            fax[1,1].scatter(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,marker=marker[trainer],color=color[trainer])         
            fax[1,1].plot(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,color=color[trainer],linewidth=2)         
    
        if len(t_mat[t_mat['Bz']    > -9990.0]) > 0:
            fax[2,1].scatter(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,marker=marker[trainer],color=color[trainer])         
            fax[2,1].plot(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,color=color[trainer],linewidth=2)         
    
    
    
        #plot observed break time
        for pax in fax.ravel():
            xoff = pd.to_timedelta('90 seconds')  
            pax.axvline(i,linewidth=3,alpha=0.5,color='purple')
            pax.axvline(i+xoff,alpha=0.5,linestyle='--',linewidth=3,color='purple')
            pax.axvline(i-xoff,alpha=0.5,linestyle='--',linewidth=3,color='purple')
           
        #plot best time for each spacecraft
        fax[1,0].legend(loc='best',frameon=False,scatterpoints=1)
    
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
    
        fax[0,0].set_xlim([i-plt_windw,i+plt_windw])
        bfig.savefig('../plots/spacecraft_events/full_res_event_{0:%Y%m%d_%H%M%S}_bigg.png'.format(i.to_pydatetime()),bbox_pad=.1,bbox_inches='tight')
    
        #close output file
        out_f.write(footer)

        #close opened plots
        plt.close(bfig)
        plt.clf()
        plt.close()
    
    #write footer and close file
    
    out_f.write(ful_ftr)
    out_f.close()




