import ftplib
import urllib
from datetime import datetime,timedelta
import itertools
import sys,os
from multiprocessing import Pool


def set_directory(craft,param):

    #ACE directories
    if ((craft == 'ace') & (param == 'orb')):
        #Updated to orbit rather than atitude 2018/01/31 J. Prchlik
        a_dir = 'ace/orbit/level_2_cdaweb/or_ssc/'
    elif ((craft == 'ace') & (param == 'mag')):
        a_dir = 'ace/mag/level_2_cdaweb/mfi_h0/'
    elif ((craft == 'ace') & (param == 'plsm')):
        a_dir = 'ace/swepam/level_2_cdaweb/swe_h0/'

    #DSCOVR directories
    elif ((craft == 'dscovr') & (param == 'orb')):
        #Updated to orbit rather than atitude 2018/01/31 J. Prchlik
        a_dir = 'dscovr/orbit/pre_or/'
    elif ((craft == 'dscovr') & (param == 'mag')):
        a_dir = 'dscovr/h0/mag/'
    elif ((craft == 'dscovr') & (param == 'plsm')):
        a_dir = 'dscovr/h1/faraday_cup/'

   #Wind directories
    elif ((craft == 'wind') & (param == 'orb')):
        #Updated to orbit rather than atitude 2018/01/31 J. Prchlik
        a_dir = 'wind/orbit/pre_or/'
    elif ((craft == 'wind') & (param == 'mag')):
        a_dir = 'wind/mfi/mfi_h2/'
    elif ((craft == 'wind') & (param == 'plsm')):
        a_dir = 'wind/swe/swe_h1/'

    #return specific directory
    return a_dir

#get list of years to look for data
def return_years(start,end):
    return range(start.year,end.year+1)

#wrapper for download files program
def download_files_wrapper(args):
    return download_files(*args)

#Actually where we download files
def grab_files(craft,param,year,start,end,ftp):

    #get subdirectory location on ftp server
    a_dir = set_directory(craft,param)
    #change ftp base directory
    ftp.cwd(a_dir)

    #list of files to download
    d_list = []
    #loop over all years and look for files
    for i in year:
        #get list of files in directory
        #Ace orbital files are not in year subdirectories so just search given directory
        if ((craft != 'ace') & (param != 'orb')):
            f_list = ftp.nlst('{0:4d}/*cdf'.format(i))
        else:
            f_list = ftp.nlst('*_{0:4d}*cdf'.format(i))
        #get just file names and check if they exist locally
        for f_name in f_list:
            #get local name for file
            l_name = f_name.split('/')[-1] 

            #output file
            o_file = '{0}/{1}/{2}'.format(craft,param,l_name)

            #check if local file exists
            l_chck = os.path.isfile(o_file) == False
            #if file does not exist or there is only 1 cdf in directory, download 
            if ((l_chck) | (len(f_list) == 1)): 
                fhandle = open(o_file,'wb')
                ftp.retrbinary('RETR {0}'.format(f_name),fhandle.write)
                fhandle.close()
     
    
#download the archive files locally
def download_files(craft,param,archive,years,start,end):

    #ftp connection
    ftp = ftplib.FTP(archive.split('/')[2],'anonymous')

    #separator to join archive string back
    sep = '/'

    #change to base directory
    ftp.cwd(sep.join(archive.split('/')[3:]))
    try:
        grab_files(craft,param,years,start,end,ftp)
    except:
        print('FTP connection failed unexpectly. Closing connection',sys.exc_info()[0],craft,param)
    #close ftp connection
    ftp.close()

#Main function to run
def main(f_types=['mag','plsm','orb'],space_c=['ace','dscovr','wind'],
         archive='ftp://cdaweb.gsfc.nasa.gov/pub/data/',start=datetime(2015,6,1),end=None,
         nproc=1):
    '''
    Python module for downloading cdf files from ftp archive to your local machine.
    
    Parameters:
    ----------
    f_types: list, optional
        List of parameters to download for ftp archive (default = ['mag','plsm','orb').
        Currently only accepts 'mag', 'plsm', and/or 'orb'.
    space_c: list, optional 
        List of spacecraft to download data for (default = ['ace','dscovr','wind']). 
        Currently only accepts 'ace', 'dscovr', and/or 'wind'.
    archive: string, optional
        String containing the base location of the ftp archive (default = 'ftp://cdaweb.gsfc.nasa.gov/pub/data/').
    start: datetime object, optional
        The start time to look for solar wind observations. Currently only uses year (Default = datetime(2015,6,1).
    end  : datetime object, optional
        The end tiem to look for solar wind observations. Currenly only uses year (Default = None).
        If end = None then program will get the current UTC year.
    nproc: int, optional
        Number of processors to you to download files (Default = 1).
   

    '''


    #if end time is not specified set now to end date
    if end == None:
        end = datetime.utcnow()

    #get all combinations of parameters and spacecraft
    comb = itertools.product(space_c,f_types)

    #year to search for data
    years = return_years(start,end)


    #create input list to loop over
    input_list = []

    #create directories and make input list
    for i in comb:
        #make subdirectories
        try:
            os.makedirs('{0}/{1}'.format(i[0],i[1]))
        except OSError:
            print('Directory Already Exists. Proceeding...')
        input_list.append([i[0],i[1],archive,years,start,end])

    #loop and download files if nproc less than 2 otherwise download in parellel
    if nproc < 2:
        for i in input_list: download_files_wrapper(i)
    else:
        pool = Pool(processes=nproc)
        outp = pool.map(download_files_wrapper,input_list)
        pool.close()
        pool.join()
