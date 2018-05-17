from spacepy import pycdf
import numpy as np
import pandas as pd
from glob import glob
import os,sys
import pandas as pd
from datetime import timedelta

from multiprocessing import Pool


def wrap_looper(inp):
    """
    Wrapper for parrallel processing

    """
    return looper(*inp)

def looper(sc1,pls,mag,orb,lstr,brchive='../cdf/'):
    """
    Function specifying which parameters to use when creating a formatted file
  
    Parameters:
    -----------
    sc1:  string
        List of spacecraft to created formatted text files for 
        (Allow inputs are 'wind','ace','dscovr','soho','themis_a','themis_b','themis_c')
    pls: boolean
        Create formatted plasma parameter file.
    mag: boolean
        Create formatted magnetic field file.
    orb: boolean
        Create formatted orbit file.
    lstr: list of strings
        String of input start dates in YYYYMMDD format 
    brchive: string,optional
        Base location of cdf archives

    """

    ##List of possible spacecraft
    scrf = ['wind','ace','dscovr','soho','themis_a','themis_b','themis_c']

    #exit cleanly if spacecraft is not in list of possible spacecraft
    if sc1 not in scrf: 
        print('You input '+sc1)
        print('Allowed inputs are:')
        print(scrf)
        print('Skipping '+sc1)
        return
   
    ##current space craft
    #sc1 = scrf[s_idx]

    
    
    #archive location of all spacecraft
    archive = brchive+'/'+sc1+'/plsm/'
    mrchive = brchive+'/'+sc1+'/mag/'
    orchive = brchive+'/'+sc1+'/orb/'

    
    
    #setup keys for particular spacecraft
    if sc1 == 'wind':
        mag_key = ['Epoch','BGSE','SPC_MODE']
        pls_key = ['Epoch','Proton_V_nonlin','Proton_Np_nonlin','Proton_W_nonlin','fit_flag']
        orb_key = ['Epoch','GSE_POS']
    if sc1 == 'ace':
        mag_key = ['Epoch','BGSEc','Q_FLAG']
        pls_key = ['Epoch','SPEED','Np','Tpr','alpha_ratio']
        orb_key = ['Epoch','XYZ_GSE']
    if sc1 == 'dscovr':
        mag_key = ['Epoch1','B1GSE','FLAG1']
        pls_key = ['Epoch','SPEED','Np','THERMAL_SPD','DQF']
        orb_key = ['Epoch','GSE_POS']

    #choose peir because it had the most data (~4s cadence for all parameters) 2018/04/19 J. Prchlik
    #choose peem because moments because it is more complete 2018/04/19 J. Prchlik
    #choose peem Removing thermal velocity for now 2018/04/19 J. Prchlik
    #choose peim because that i stands for ion and e stands for electon 2018/04/20 J. Prchlik
    if sc1 == 'themis_c':
        mag_key = ['thc_peim_time','thc_peim_mag']
        pls_key = ['thc_peim_time','thc_peim_velocity_gse','thc_peim_density','thc_peim_t3_mag','thc_peim_data_quality']
        orb_key = ['Epoch','XYZ_GSE']
    if sc1 == 'themis_b':
        mag_key = ['thb_peim_time','thb_peim_mag']
        pls_key = ['thb_peim_time','thb_peim_velocity_gse','thb_peim_density','thb_peim_t3_mag','thb_peim_data_quality']
        orb_key = ['Epoch','XYZ_GSE']
    if sc1 == 'themis_a':
        mag_key = ['tha_peim_time','tha_peim_mag']
        pls_key = ['tha_peim_time','tha_peim_velocity_gse','tha_peim_density','tha_peim_t3_mag','tha_peim_data_quality']
        orb_key = ['Epoch','XYZ_GSE']

    #Get magnetic and plasma cdf files
    t_fpls = glob(archive+'*cdf')
  
    #different name format for wind observations
    if sc1 == 'wind': 
         t_fmag = glob(mrchive+'*h2*cdf')
    else:
         t_fmag = glob(mrchive+'*h0*cdf')
    t_forb = glob(orchive+'*or*cdf')

   
    #Use combined moment data because 4 second cadence is good enough time resolution 2018/04/20 J. Prchlik
    if 'themis' in sc1: t_fmag = glob(archive+'*mom*cdf')

    #Only use files in time range
    fpls = [ s for s in t_fpls for d in lstr if d in s]
    fmag = [ s for s in t_fmag for d in lstr if d in s]
    #check for orbital files only that  have the year
    if sc1 == 'ace':
        cstr = np.unique([d[:4] for d in lstr]) 
        forb = [ s for s in t_forb for d in cstr if d in s]
    #Or 1 per month
    elif 'themis' in sc1:
        cstr = np.unique([d[:6] for d in lstr]) 
        forb = [ s for s in t_forb for d in cstr if d in s]
    else:
        forb = [ s for s in t_forb for d in lstr if d in s]
    


    #list dataframe to return
    ret_df = {}
    #add spacecraft identification
    ret_df['craft'] = sc1
    
    #convert to textfile
    #Commented to fix time error J. Prchlik 2017/11/14
    #creating logic switches 
    if pls:
        ret_df['pls'] = cdf_to_pandas(fpls,pls_key,sc1,'pls')
    ##commented out J. Prchlik 2017/11/14 to fix wrong Vth in ACE
    if mag:
        ret_df['mag'] = cdf_to_pandas(fmag,mag_key,sc1,'mag')
    #Add orbital files 2018/01/31 J. Prchlik
    if orb: 
        ret_df['orb'] = cdf_to_pandas(forb,orb_key,sc1,'orb')

    return ret_df

#function to create pandas dataframe
def cdf_to_pandas(f_list,keys,craft,context):
    """
    Function which creates a formatted pandas dataframe for a given spacecraft and parameter (i.e. mag. field, plasma, or orbit location)
   
    Parameters:
    ----------
    flist: list
        Full path list of cdf files
    keys: list
        Keys to retrieve from the CDF file 
    craft:  string
        Spacecraft to created formatted text files for
    context: string
        Whether the keys refer to magnetic field, plasma, or orbital parameters

    """

    #boltzman constant and proton mass for conversion of temp to speed
    amu = 1.660538921e-27
    mp  = 1.00727647*amu
    kb  = 1.3906488e-23

    #oupt variable in header
    if context =='pls':
        header = ['Time','SPEED','Np','Vth','DQF']
    elif context == 'mag':
        header = ['Time','Bx','By','Bz','DQF']
    elif context == 'orb':
        header = ['Time','GSEx','GSEy','GSEz']


    #create initial data frame
    tab = pd.DataFrame(columns=header)

    #loop over all files and write text file
    for i in f_list:

        #read cdffile
        cdf = pycdf.CDF(i)


        #######Plasma Parameters#########
        if ((context == 'pls') & (craft == 'wind')):
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,float(cdf[keys[1]][k]),float(cdf[keys[2]][k]),float(cdf[keys[3]][k]),int(cdf[keys[4]][k])]
            #Switched to effecienct array creation 2018/05/17 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...],cdf[keys[2]][...],cdf[keys[3]][...],cdf[keys[4]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'pls') & (craft == 'dscovr')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k]),int(cdf[keys[4]][k])]
            #Switched to effecienct array creation 2018/05/17 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],SPEED,cdf[keys[2]][...],cdf[keys[3]][...],cdf[keys[4]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'pls') & (craft == 'ace')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            Vth   = 1.E-3*np.sqrt(2.*kb/mp*cdf[keys[3]][...]) #convert Thermal Temp to Speed
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,SPEED[k],float(cdf[keys[2]][k]),float(Vth[k]),0]
            #Switched to effecienct array creation 2018/05/17 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],SPEED,cdf[keys[2]][...],Vth,np.zeros(len(Vth))]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'pls') & ('themis' in craft)):
            SPEED = np.sqrt(np.sum(cdf[keys[1]][...]**2,axis=1))
            epoch = pd.to_timedelta(cdf[keys[0]][...],unit='s')+pd.to_datetime('1970/01/01 00:00:00' )
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([epoch,SPEED,cdf[keys[2]][...],cdf[keys[3]][...][:,0],cdf[keys[4]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 



        #######MAG fields#########
        elif ((context == 'mag') & (craft == 'wind')):
            #decrease the wind cadence 10 s in magfield
            loopers = range(0,len(cdf[keys[0]][...]),90) 
            #Update to do without looping  2018/05/17 J. Prchlik
            #for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k][0],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
            temp = pd.DataFrame(np.array([cdf[keys[0]][...].ravel()[loopers],cdf[keys[1]][...][:,0][loopers],cdf[keys[1]][...][:,1][loopers],cdf[keys[1]][...][:,2][loopers],cdf[keys[2]][...].ravel()[loopers]]).T,columns=header)
            #temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2],cdf[keys[2]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'mag') & (craft == 'dscovr')):
            #decrease the wind cadence 10 s in magfield
            #Update to do without looping  2018/05/17 J. Prchlik
            #loopers = range(0,len(cdf[keys[0]][...]),10) 
            #for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2],cdf[keys[2]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'mag') & (craft == 'ace')):
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
            #Update to do without looping  2018/05/17 J. Prchlik
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2],cdf[keys[2]][...]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'mag') & ('themis' in craft)):
            epoch = pd.to_timedelta(cdf[keys[0]][...],unit='s')+pd.to_datetime('1970/01/01 00:00:00' ) #Not corrected for leap seconds
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([epoch,cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2],np.zeros(len(epoch))]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 


        #######Orbital Parameters#########
        #Added orbital files 2018/01/31 J. Prchlik
        elif ((context == 'orb') & (craft == 'wind')):
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'orb') & (craft == 'dscovr')):
            #only get orbit every 10 minutes
            #loopers = range(0,len(cdf[keys[0]][...]),10) 
            #for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'orb') & (craft == 'ace')):
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 
        elif ((context == 'orb') & ('themis' in craft)):
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
            #Switched to effecienct array creation 2018/05/03 J. Prchlik
            temp = pd.DataFrame(np.array([cdf[keys[0]][...],cdf[keys[1]][...][:,0],cdf[keys[1]][...][:,1],cdf[keys[1]][...][:,2]]).T,columns=header)
            tab = tab.append(temp,ignore_index=True) 



        #close cdf file
        cdf.close()
        #print('Completed {0}'.format(i))
 

    return tab
    #close output file
    ####SWITCH TO PANDAS Tabling 2018/01/26 (J. Prchlik)
    #Only do if there is a reason to write out a given file (2018/01/29) J. Prchlik
    #out_fil.close()
    #if write_out:
    #    tab.to_csv(out_fil,index=None,sep=' ')
    #    tab.drop_duplicates(subset='Time',keep='last',inplace=True)
    #    tab.fillna(-9999.9,inplace=True)
    #    tab['Time'] = pd.to_datetime(tab['Time'])
    #    tab.sort_values('Time',inplace=True)
    #    #drop if we some how get duplicates
    #    tab.drop_duplicates(subset='Time',keep='last',inplace=True)
    #    tab.to_csv(out_fil,index=None,sep=' ')


def day_list(sday,eday):
    """
    Returns a list of days including the start and end days

    Parameters
    ----------
    sday: datetime object
        Starting date to read in cdf files
    eday: datetime object
        Ending date to read in cdf files

    Returns
    -------
    dates: list
        list of days from sday to eday
    """
    dates = [(sday+timedelta(n)).strftime('%Y%m%d')  for n in range(int((eday-sday).days+1))]
    return dates


def main(sday,eday,scrf=['wind','ace','dscovr','soho','themis_a','themis_b','themis_c'],nproc=1,pls=True,mag=True,orb=True):
    """
    Python module for formatting cdf files downloaded via get_cdf_files (up one directory) into text files.

    Parameters
    ----------
    stime: datetime object
        Starting date to read in cdf files
    etime: datetime object
        Ending date to read in cdf files
    scrf:  list,optional
        List of spacecraft to created formatted text files for (default= ['wind','ace','dscovr','soho','themis_a','themis_b','themis_c'])
    nproc: int, optional
        Number of processors used to format files. Can be up to 1 processor per spacecraft in scrf (default = 1).
    pls: boolean, optional
        Create formatted plasma parameter file (Default = True).
    mag: boolean, optional
        Create formatted magnetic field file (Default = True).
    orb: boolean, optional
        Create formatted orbit file (Default = True).

    Returns:
    --------
    outdict: dictionary
        Dictionary of output parameters
    

    Example:
    -------
    import load_cdf_files as lcf
    from datetime import datetime

    stime = datetime(2017,01,06)
    etime = datetime(2017,01,10)

    out = lcf.main(stime,etime,scrf=['wind','ace','dscovr','themis_b'],pls=True,mag=True,orb=True,nproc=4) 
    out = lcf.main(stime,etime,scrf=['ace'],pls=True,mag=True,orb=True) 

    """

    lday = day_list(sday,eday)
    #dictionary output
    outdict = {}
    #Do in parallel per spacecraft
    if nproc > 1:
        #Add arguments to spacecraft name        
        arg_scrf =[]
        for i in scrf: arg_scrf.append([i,pls,mag,orb,lday])

        pool = Pool(processes=nproc)
        out  = pool.map(wrap_looper,arg_scrf)
        pool.close()
        pool.join()

        #store in larger dictionary
        for i in out:
            if i is not None:
                outdict[i['craft']] = i 
    #Just loop if given 1 processor
    else:
        for i in scrf: 
            outdict[i] = looper(i,pls,mag,orb,lday)

    return outdict

    
