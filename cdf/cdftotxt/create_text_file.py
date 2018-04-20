from spacepy import pycdf
import numpy as np
import pandas as pd
from glob import glob
import os,sys
import pandas as pd

from multiprocessing import Pool


def wrap_looper(inp):
    """
    Wrapper for parrallel processing

    """
    return looper(*inp)

def looper(sc1,pls,mag,orb):
    """
    Function specifying which parameters to use when creating a formatted file
  
    Parameters:
    -----------
    sc1:  string
        List of spacecraft to created formatted text files for
    pls: boolean
        Create formatted plasma parameter file.
    mag: boolean
        Create formatted magnetic field file.
    orb: boolean
        Create formatted orbit file.

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
    archive='..'#set output cdf location
    archive = archive+'/'+sc1+'/plsm/'
    mrchive='..'#set output cdf location
    mrchive = mrchive+'/'+sc1+'/mag/'
    orchive='..'#set output cdf location
    orchive = orchive+'/'+sc1+'/orb/'

    
    
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
    fpls = glob(archive+'*cdf')
    fmag = glob(mrchive+'*h0*cdf')
    forb = glob(orchive+'*or*cdf')
   
    #Use combined moment data because 4 second cadence is good enough time resolution 2018/04/20 J. Prchlik
    if 'themis' in sc1: fmag = glob(archive+'*mom*cdf')
    
    #convert to textfile
    #Commented to fix time error J. Prchlik 2017/11/14
    #creating logic switches 
    if pls: cdf_to_text(fpls,pls_key,sc1,'pls')
    ##commented out J. Prchlik 2017/11/14 to fix wrong Vth in ACE
    if mag: cdf_to_text(fmag,mag_key,sc1,'mag')
    #Add orbital files 2018/01/31 J. Prchlik
    if orb: cdf_to_text(forb,orb_key,sc1,'orb')

#function to create pandas dataframe
def cdf_to_text(f_list,keys,craft,context):
    """
    Function which creates a formatted file for a given spacecraft and parameter (i.e. mag. field, plasma, or orbit location)
   
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


    #output header
    out_hdr = '{0}                          {1}               {2}              {3}             {4}\n'
    #output format
    out_fmt = '{0:%Y/%m/%dT%H:%M:%S}  {2:15.2f}  {3:15.2f}  {4:15.2f}             {1}\n'

    #get name of output file
    out_fil = '{0}_{1}_2015_2017_formatted.txt'.format(craft,context)

    #see if output file already exists
    out_chk = os.path.isfile(out_fil)

    #if file exists read in the file
    if out_chk: 
        tab = pd.read_table(out_fil,delim_whitespace=True)
        #create datetime objects from time
        tab['Time'] = pd.to_datetime(tab['Time'])
        #setup index
        #day text 
        day_txt = tab.Time.dt.strftime('%Y%m%d')

        #tab.set_index(tab.time_dt,inplace=True)

    #create new table
    else:
        tab = pd.DataFrame(columns=header)
        day_txt = np.array(['17760704','19580102'])
    #out_fil = open('{0}_{1}_2017_2017_formatted.txt'.format(craft,context),'w')

    ###write header
    #out_fil.write(out_hdr.format(*header))

    #Wether or not to write out a new file at the end
    write_out = False

    #loop over all files and write text file
    for i in f_list:

        #get day of observation
        day = i.split('_')[-2]
 
        #check if day is already in file
        day_chk = (day_txt == day).any()
   
        #if day already exists continue in loop
        if day_chk: continue

        #If you get this far write out a new file because it means there is a new cdf
        write_out = True
     
        #read cdffile
        cdf = pycdf.CDF(i)


        #loop through cdf and write
        ####SWITCH TO PANDAS Tabling 2018/01/26 (J. Prchlik)
        ####if ((context == 'pls') & (craft == 'wind')):
        ####    for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),float(cdf[keys[1]][k]),float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        ####elif ((context == 'pls') & (craft == 'dscovr')):
        ####    SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
        ####    for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        ####elif ((context == 'pls') & (craft == 'ace')):
        ####    SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
        ####    Vth   = 1.E-3*np.sqrt(2.*kb/mp*cdf[keys[3]][...]) #convert Thermal Temp to Speed
        ####    for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),SPEED[k],float(cdf[keys[2]][k]),float(Vth[k])))
        ####elif ((context == 'mag') & (craft == 'wind')):
        ####    #decrease the wind cadence 10 s in magfield
        ####    loopers = range(0,len(cdf[keys[0]][...]),90) 
        ####    for k in loopers: out_fil.write(out_fmt.format(cdf[keys[0]][k][0],int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        ####elif ((context == 'mag') & (craft == 'dscovr')):
        ####    #decrease the wind cadence 10 s in magfield
        ####    loopers = range(0,len(cdf[keys[0]][...]),10) 
        ####    for k in loopers: out_fil.write(out_fmt.format(cdf[keys[0]][k],int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        ####elif ((context == 'mag') & (craft == 'ace')):
        ####    #for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        ####    #Hacked for k1 observations
        ####    for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(0),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        if ((context == 'pls') & (craft == 'wind')):
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,float(cdf[keys[1]][k]),float(cdf[keys[2]][k]),float(cdf[keys[3]][k]),int(cdf[keys[4]][k])]
        elif ((context == 'pls') & (craft == 'dscovr')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k]),int(cdf[keys[4]][k])]
        elif ((context == 'pls') & (craft == 'ace')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            Vth   = 1.E-3*np.sqrt(2.*kb/mp*cdf[keys[3]][...]) #convert Thermal Temp to Speed
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,SPEED[k],float(cdf[keys[2]][k]),float(Vth[k]),0]
        elif ((context == 'pls') & ('themis' in craft)):
            SPEED = np.sqrt(np.sum(cdf[keys[1]][...]**2,axis=1))
            epoch = pd.to_timedelta(cdf[keys[0]][...],unit='s')+pd.to_datetime('1970/01/01 00:00:00' )
            #Added radial thermal velocity measurement
            for k,j in enumerate(epoch): tab.loc[len(tab)] = [j,SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k][0]),int(cdf[keys[4]][k])]
        elif ((context == 'mag') & (craft == 'wind')):
            #decrease the wind cadence 10 s in magfield
            loopers = range(0,len(cdf[keys[0]][...]),90) 
            for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k][0],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
        elif ((context == 'mag') & (craft == 'dscovr')):
            #decrease the wind cadence 10 s in magfield
            loopers = range(0,len(cdf[keys[0]][...]),10) 
            for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
        elif ((context == 'mag') & (craft == 'ace')):
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2],int(cdf[keys[2]][k])]
        elif ((context == 'mag') & ('themis' in craft)):
            epoch = pd.to_timedelta(cdf[keys[0]][...]-11.79,unit='s')+pd.to_datetime('1970/01/01 00:00:00' ) #Not corrected for leap seconds
            #Cut to ~10s 
            #Removed mag field cut because I am using a 4 s cadence combined file 2018/04/20 J. Prchlik
            for k,j in enumerate(epoch): tab.loc[len(tab)] = [j,cdf[keys[1]][k][0],cdf[keys[1]][k][1],cdf[keys[1]][k][2],0]
            #Hacked for k1 observations
            #Undone 2018/01/26 (J. prchlik)
            #for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,int(0),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
        #Added orbital files 2018/01/31 J. Prchlik
        elif ((context == 'orb') & (craft == 'wind')):
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
        elif ((context == 'orb') & (craft == 'dscovr')):
            #only get orbit every 10 minutes
            loopers = range(0,len(cdf[keys[0]][...]),10) 
            for k in loopers: tab.loc[len(tab)] = [cdf[keys[0]][k],(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
        elif ((context == 'orb') & (craft == 'ace')):
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]
        elif ((context == 'orb') & ('themis' in craft)):
            for k,j in enumerate(cdf[keys[0]][...]): tab.loc[len(tab)] = [j,(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]]



        #close cdf file
        cdf.close()
        print('Completed {0}'.format(i))
 

    #close output file
    ####SWITCH TO PANDAS Tabling 2018/01/26 (J. Prchlik)
    #Only do if there is a reason to write out a given file (2018/01/29) J. Prchlik
    #out_fil.close()
    if write_out:
        tab.fillna(-9999.9,inplace=True)
        tab['Time'] = pd.to_datetime(tab['Time'])
        tab.sort_values('Time',inplace=True)
        #drop if we some how get duplicates
        tab.drop_duplicates(subset='Time',keep='last',inplace=True)
        tab.to_csv(out_fil,index=None,sep=' ')


#loop over spacecraft
#s_idx = 0
#fix index error in DSCOVR and Wind Mag. Field
#Run 2015-2017 observations
ids = [0,1,2]
ids = [4,5]

def main(scrf=['wind','ace','dscovr','soho','themis_a','themis_b','themis_c'],nproc=1,pls=True,mag=True,orb=True):
    """
    Python module for formatting cdf files downloaded via get_cdf_files (up one directory) into text files.

    Parameters
    ----------
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

    Example:
    -------
    import create_text_file as ctf
    cft.main(scrf=['themis_b'],pls=True,mag=True,orb=False) 

    """

    #Do in parallel per spacecraft
    if nproc > 1:
        #Add arguments to spacecraft name        
        arg_scrf =[]
        for i in scrf: arg_scrf.append([i,pls,mag,orb])

        pool = Pool(processes=nproc)
        out  = pool.map(wrap_looper,arg_scrf)
        pool.close()
        pool.join()
    else:
        for i in scrf: looper(i,pls,mag,orb)
    
