import ftplib
import urllib
from datetime import datetime,timedelta
import itertools


def set_directory(craft,param):


    #ACE directories
    if ((craft == 'ace') & (param == 'orb')):
        a_dir = 'ace/orbit/level_2_cdaweb/def_at/'
    elif ((craft == 'ace') & (param == 'mag')):
        a_dir = 'ace/orbit/level_2_cdaweb/def_at/'
    elif ((craft == 'ace') & (param == 'plsm')):
        a_dir = 'ace/orbit/level_2_cdaweb/def_at/'

    #DSCOVR directories
    elif ((craft == 'dscovr') & (param == 'orb')):
        a_dir = 'ace/orbit/level_2_cdaweb/def_at/'



def main(f_types=['mag','plsm','orb'],space_c=['ace','dscovr','wind'],
         archive='ftp://cdaweb.gsfc.nasa.gov/pub/data/',start=datetime(2015,6,1),end=None,
         bdir='ftp://cdaweb.gsfc.nasa.gov/pub/data/ace/orbit/level_2_cdaweb/def_at/'):


    #if end time is not specified set now to end date
    if end == None:
        end = datetime.utcnow()

    #get all combinations of parameters and spacecraft
    comb = itertools.product(space_c,f_types)
