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



