from spacepy import pycdf
import numpy as np
import pandas as pd
from glob import glob

from multiprocessing import Pool


def looper(s_idx):
    #List of possible spacecraft
    scrf = ['wind','ace','dscovr','soho']
    #current space craft
    sc1 = scrf[s_idx]

    
    
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
    if sc1 == 'ace':
        mag_key = ['Epoch','BGSEc','Q_FLAG']
        pls_key = ['Epoch','SPEED','Np','Tpr','alpha_ratio']
    if sc1 == 'dscovr':
        mag_key = ['Epoch1','B1GSE','FLAG1']
        pls_key = ['Epoch','SPEED','Np','THERMAL_SPD','DQF']

    #Get magnetic and plasma cdf files
    fpls = glob(archive+'*20170[8-9]*cdf')
    fmag = glob(mrchive+'*k1*cdf')
    
    #convert to textfile
    #Commented to fix time error J. Prchlik 2017/11/14
    #just ace currupted magnetic field observations
    #cdf_to_text(fpls,pls_key,sc1,'pls')
    #commented out J. Prchlik 2017/11/14 to fix wrong Vth in ACE
    cdf_to_text(fmag,mag_key,sc1,'mag')

#function to create pandas dataframe
def cdf_to_text(f_list,keys,craft,context):

    #boltzman constant and proton mass for conversion of temp to speed
    amu = 1.660538921e-27
    mp  = 1.00727647*amu
    kb  = 1.3906488e-23

    #oupt variable in header
    if context =='pls':
        header = ['Time','SPEED','Np','Vth','DQF']
    elif context == 'mag':
        header = ['Time','Bx','By','Bz','DQF']


    #output header
    out_hdr = '{0}                          {1}               {2}              {3}             {4}\n'
    #output format
    out_fmt = '{0:%Y/%m/%dT%H:%M:%S}  {2:15.2f}  {3:15.2f}  {4:15.2f}             {1}\n'

    #output text file
    out_fil = open('{0}_{1}_2017_2017_formatted.txt'.format(craft,context),'w')


    #write header
    out_fil.write(out_hdr.format(*header))

   

    #loop over all files and write text file
    for i in f_list:

        #read cdffile
        cdf = pycdf.CDF(i)


        #loop through cdf and write
        if ((context == 'pls') & (craft == 'wind')):
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),float(cdf[keys[1]][k]),float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        elif ((context == 'pls') & (craft == 'dscovr')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        elif ((context == 'pls') & (craft == 'ace')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            Vth   = 1.E-3*np.sqrt(2.*kb/mp*cdf[keys[3]][...]) #convert Thermal Temp to Speed
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),SPEED[k],float(cdf[keys[2]][k]),float(Vth[k])))
        elif ((context == 'mag') & (craft == 'wind')):
            #decrease the wind cadence 10 s in magfield
            loopers = range(0,len(cdf[keys[0]][...]),90) 
            for k in loopers: out_fil.write(out_fmt.format(cdf[keys[0]][k][0],int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        elif ((context == 'mag') & (craft == 'dscovr')):
            #decrease the wind cadence 10 s in magfield
            loopers = range(0,len(cdf[keys[0]][...]),10) 
            for k in loopers: out_fil.write(out_fmt.format(cdf[keys[0]][k],int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        elif ((context == 'mag') & (craft == 'ace')):
            #for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[2]][k]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
            #Hacked for k1 observations
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(0),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))



        #close cdf file
        cdf.close()
 

    #close output file
    out_fil.close()


#loop over spacecraft
#s_idx = 0
#fix index error in DSCOVR and Wind Mag. Field
#Run 2015-2017 observations
ids = [0,1,2]

#Do in parallel
#pool = Pool(processes=3)
#out  = pool.map(looper,ids)
#pool.close()
#pool.join()

#just ace to fix wrong thermal speed
#just ace currupted magnetic field observations
looper(1)

#for s_idx in range(3): looper(s_idx)
