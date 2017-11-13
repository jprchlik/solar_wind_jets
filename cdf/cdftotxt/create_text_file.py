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
    archive='/Volumes/Pegasus/jprchlik/dscovr/solar_wind_events/cdf'#set output cdf location
    archive = archive+'/'+sc1+'/plsm/'
    mrchive='/Volumes/Pegasus/jprchlik/dscovr/solar_wind_events/cdf'#set output cdf location
    mrchive = mrchive+'/'+sc1+'/mag/'
    
    
    #setup keys for particular spacecraft
    if sc1 == 'wind':
        mag_key = ['Epoch','BGSE','FLAG1_I']
        pls_key = ['Epoch','Proton_V_nonlin','Proton_Np_nonlin','Proton_W_nonlin','fit_flag']
    if sc1 == 'ace':
        mag_key = ['Epoch','BGSEc','Q_FLAG']
        pls_key = ['Epoch','SPEED','Np','Vp','alpha_ratio']
    if sc1 == 'dscovr':
        mag_key = ['Epoch1','B1GSE','FLAG1']
        pls_key = ['Epoch','SPEED','Np','THERMAL_SPD','DQF']

    #Get magnetic and plasma cdf files
    fpls = glob(archive+'*cdf')
    fmag = glob(mrchive+'*cdf')
    
    #convert to textfile
    cdf_to_text(fpls,pls_key,sc1,'pls')
    cdf_to_text(fmag,mag_key,sc1,'mag')

#function to create pandas dataframe
def cdf_to_text(f_list,keys,craft,context):


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
    out_fil = open('{0}_{1}_formatted.txt'.format(craft,context),'w')


    #write header
    out_fil.write(out_hdr.format(*header))

    #loop over all files and write text file
    for i in f_list:

        #read cdffile
        cdf = pycdf.CDF(i)


        #loop through cdf and write
        if ((context == 'pls') & (craft == 'wind')):
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),float(cdf[keys[1]][k]),float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        elif ((context == 'pls') & (craft != 'wind')):
            SPEED = np.sqrt(np.sum(cdf['V_GSE'][...]**2,axis=1))
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[4]][k]),SPEED[k],float(cdf[keys[2]][k]),float(cdf[keys[3]][k])))
        elif ((context == 'mag') & (craft == 'wind')):
            #decrease the wind cadence 10 s in magfield
            for k,j in enumerate(cdf[keys[0]][...][::90]): out_fil.write(out_fmt.format(j[0],int(cdf[keys[2]][0]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        elif ((context == 'mag') & (craft == 'dscovr')):
            #decrease the wind cadence 10 s in magfield
            for k,j in enumerate(cdf[keys[0]][...][::10]): out_fil.write(out_fmt.format(j,int(cdf[keys[2]][0]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))
        elif ((context == 'mag') & (craft == 'ace')):
            for k,j in enumerate(cdf[keys[0]][...]): out_fil.write(out_fmt.format(j,int(cdf[keys[2]][0]),(cdf[keys[1]][k][0]),cdf[keys[1]][k][1],cdf[keys[1]][k][2]))



        #close cdf file
        cdf.close()
 

    #close output file
    out_fil.close()


#loop over spacecraft
#s_idx = 0

#Do in parallel
pool = Pool(processes=3)
out  = pool.map(looper,range(3))
pool.close()
pool.join()


#for s_idx in range(3): looper(s_idx)
