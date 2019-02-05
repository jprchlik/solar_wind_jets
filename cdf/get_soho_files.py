import datetime as dt
import glob
import numpy as np
import zipfile

try:
    #for python 3.0 or later
    from urllib.request import urlopen
except ImportError:
    #Fall back to python 2 urllib2
    from urllib2 import urlopen

import pandas as pd
import os

def format_file(fname):

    #Read lines in file
    x = open(fname).readlines()

    #Find line with formatting keywords
    while test:
         m = x[j].strip()
         if 'FORMAT' in m:
             test = False
         j+=1
 

    

#years of interest
start_year= 2016
cur_year = dt.datetime.utcnow().year

#Location of SOHO/PM achives per year at 30s cadence
soho_arch = "http://l1.umd.edu/{0:4d}_CELIAS_Proton_Monitor_30s.zip"
#Naming convention of output file
outf = '{0:4d}.txt'

i = start_year

while i <= cur_year:
    #setup filename of downloaded zip file
    fname = 'soho/'+soho_arch.split('/')[-1].format(i) 

    #download 30s data from the SOHO archive
    try:
        res = urlopen(soho_arch.format(i))
    #Go to next year if a year is missing
    except:
        i+= 1
        continue

    dat = res.read()
    fo = open(fname,'w')
    fo.write(dat)
    fo.close()

    #Unzip the downloaded file
    zip_ref = zipfile.ZipFile(fname, 'r')
    zip_ref.extractall(fname.split('/')[0])
    zip_ref.close()

    #name of the unzipped file
    uzip_file = outf.format(i)

    #create pandas dataframe of input file
    data = pd.read_csv('soho/'+uzip_file,sep='\s+',skiprows=28,header=0)

    #Add missing magentic field observations
    data['Bx'] = -9999.9
    data['By'] = -9999.9
    data['Bz'] = -9999.9

    #Just Data Quality flag to 0 (i.e., good)
    data['DQF'] = 0


    #Convert more standard DOY
    data['dt'] = pd.to_datetime('{0:4d}:'.format(i)+data['DOY:HH:MM:SS'],format='%Y:%j:%H:%M:%S')
    data['Time'] = data.dt.dt.strftime('%Y/%m/%dT%H:%M:%S')


    #Rename position columns
    data.rename(columns={'GSE_X': 'X', 'GSE_Y': 'Y', 'GSE_Z':'Z'}, inplace=True)
    

    #Columns to Write
    write_col = ['Time','DQF','SPEED','Np','Vth','X','Y','Z','Bx','By','Bz']

    #write output file
    data[write_col].to_csv('soho/soho_h1_fc_{0:4d}.txt'.format(i),sep=' ',index=False)

    i+= 1