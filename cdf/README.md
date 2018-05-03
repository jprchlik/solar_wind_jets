FILES TO USE from cdaweb (ftp = ftp://cdaweb.gsfc.nasa.gov/pub/data/)

The work horse!!!!

cdftotxt
--------
Directory containing text files created from the cdf files and the python program to make the text files.

get_cdf_files.py downloads cdf files from cdaweb.


    '''
    Python module for downloading cdf files from ftp archive to your local machine.
    
    Parameters:
    ----------
    f_types: list, optional
        List of parameters to download for ftp archive (default = ['mag','plsm','orb']).
        Currently only accepts 'mag', 'plsm', and/or 'orb'.
    space_c: list, optional 
        List of spacecraft to download data for (default = ['ace','dscovr','wind'] but may also take 'themis_a', 'themis_b', and 'themis_c']). 
    archive: string, optional
        String containing the base location of the ftp archive (default = 'ftp://cdaweb.gsfc.nasa.gov/pub/data/').
    start: datetime object, optional
        The start time to look for solar wind observations. Currently only uses year (Default = datetime(2015,6,1).
    end  : datetime object, optional
        The end tiem to look for solar wind observations. Currenly only uses year (Default = None).
        If end = None then program will get the current UTC year.
    nproc: int, optional
        Number of processors to you to download files (Default = 1).


    Example:
    --------
    import get_cdf_files as gcf
    from datetime import datetime
    stime = datetime(2016,12,20)
    etime = datetime(2016,12,22)

    gcf.main(space_c=['themis_c'],start=stime,end=etime)
   

    '''




ACE:

AC_H0_MFI (ace/mag/ 2016/01/01-2017/07/28)
H0 - ACE Magnetic Field 16-Second Level 2 Data - N. Ness (Bartol Research Institute)
Available dates: 1997/09/02 00:00:12 - 2017/07/28 23:59:51

AC_H0_SWE (ace/plsm 2016/01/01-2017/08/24)

ACE/SWEPAM Solar Wind Experiment 64-Second Level 2 Data - D. J. McComas (SWRI)
Available dates: 1998/02/04 00:00:31 - 2017/08/24 23:59:05
(Continuous coverage not guaranteed - check the inventory graph for coverage)

DSCOVR:

DSCOVR_H0_MAG (dscovr/mag 2016/01/01-2017/10/05)

DSCOVR Fluxgate Magnetometer 1-sec Definitive Data - A. Koval (UMBC, NASA/GSFC)
Available dates: 2015/06/08 00:00:00 - 2017/10/05 23:59:59
(Continuous coverage not guaranteed - check the inventory graph for coverage)

WIND:
WI_H2_MFI (wind/mag 2016/01/01-2017/11/05 )

Wind Magnetic Fields Investigation, High-resolution Definitive Data - A. Szabo (NASA/GSFC)
Available dates: 1994/11/13 15:50:26 - 2017/11/05 23:59:59
(Continuous coverage not guaranteed - check the inventory graph for coverage)

WI_H1_SWE (wind/plsm 2016/01/01-2017/09/20) [Got from  jprchlik@rad.cfa.harvard.edu:/crater/observatories/wind/swe/nssdc_cdf/wi_h1_swe_201[6-7].cdf )

Wind Solar Wind Experiment, 92-sec Solar Wind Alpha and Proton Anisotropy Analysis - K. Ogilvie (NASA GSFC)
Available dates: 1994/11/17 19:50:45 - 2017/09/20 23:58:17