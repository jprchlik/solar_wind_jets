Convert the local archive of cdf files into an easy to use txt file.

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
    ctf.main(scrf=['themis_b'],pls=True,mag=True,orb=False) 
    """