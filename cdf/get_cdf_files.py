import ftplib
import urllib
from datetime import datetime,timedelta
import itertools


def set_directory(craft,param):

    #ACE directories
    if ((craft == 'ace') & (param == 'orb')):
        a_dir = 'ace/orbit/level_2_cdaweb/def_at/'
    elif ((craft == 'ace') & (param == 'mag')):
        a_dir = 'ace/mag/level_2_cdaweb/mfi_h0/'
    elif ((craft == 'ace') & (param == 'plsm')):
        a_dir = 'ace/swepam/level_2_cdaweb/swe_h0/'

    #DSCOVR directories
    elif ((craft == 'dscovr') & (param == 'orb')):
        a_dir = 'dscovr/orbit/def_at/'
    elif ((craft == 'dscovr') & (param == 'mag')):
        a_dir = 'dscovr/h0/mag/'
    elif ((craft == 'dscovr') & (param == 'plsm')):
        a_dir = 'dscovr/h1/faraday_cup/'

   #Wind directories
    elif ((craft == 'wind') & (param == 'orb')):
        a_dir = 'wind/orbit/pre_at/'
    elif ((craft == 'wind') & (param == 'mag')):
        a_dir = 'wind/mfi/mfi_h2/'
    elif ((craft == 'wind') & (param == 'plsm')):
        a_dir = 'wind/swe/swe_h1/'

    #return specific directory
    return a_dir


def main(f_types=['mag','plsm','orb'],space_c=['ace','dscovr','wind'],
         archive='ftp://cdaweb.gsfc.nasa.gov/pub/data/',start=datetime(2015,6,1),end=None,
         bdir='ftp://cdaweb.gsfc.nasa.gov/pub/data/'):


    #if end time is not specified set now to end date
    if end == None:
        end = datetime.utcnow()

    #get all combinations of parameters and spacecraft
    comb = itertools.product(space_c,f_types)
