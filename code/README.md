Directory containing programs to analyze solar wind observations.

bin_shock_time.py
----------------
Old, just returned discontinuities as a function of time.

find_shocks.py
--------------
Old, Bokeh plot to find on which parameter to use for predicting solar wind observations.

loop_train_shock (all).py
--------------------
Old, Used to try to find best parameters for find discontinuities in solar wind.

match_mikes_big_events.py
------------------------
Compare planar solutions I derive to the values Mike derives.

matching_event.py
----------------
Old, Match L1 spacecraft time for 1 instance over multiple events and write out to html file. Used load plasma data (under sampled)

matching_event_full_res.py
----------------
Match L1 spacecraft time for 1 instance over multiple events and write out to html file. Uses cdftotxt files (full resolution).


    '''
    Function which finds solar wind events in a specific spacecraft. Then the program does a Chi^2 minimization to find the event
    in the other spacecraft. In doing so it creates a series of plots and a html table to summarize the results.
    Usage
    ----------
    Example:
    python> import matching_events_full_res as fr
    python> import pandas as pd
    python> fr.main(nproc=1,p_val=0.950,use_chisq=False,use_discon=False,use_dtw=True ,rgh_chi_t=pd.to_timedelta('50 minutes'),refine=False,start_t='2016/06/04',end_t='2017/09/14',plot=True ,center=True)
    Parameters
    -----------
    craft: list, optional
        List of spacecraft to analyze for events. The first spacecraft in the list finds the events and is the reference point
        for the Chi^2 minimization (Default = ['Wind','DSCOVR','ACE','SOHO']).
    col  : list, optional
        List of matplotlib colors for plotting spacecraft points. The color list index corresponds to the index in craft
        (Default = ['blue','black','red','teal']).
    mar  : list, optional
        List of matplotlib markers for plotting spacecraft points. The marker list index corresponds to the index in craft
        (Default = ['D','o','s','<']).
    use_craft : boolean, optional
        Use spacecraft positions to restrict finding range (Default = False). It turns out a lot of events are not radial 
        therefore use_craft is no useful. 
    use_chisq : boolean, optional
        Use Chi^2 minimization to find corresponding events between spacecraft. The Chi^2 minimum value relates to a given 
        spacecraft and the first spacecraft in the craft list (Default = True).
    use_discon: boolean, optional
        Use Chi^2 minimization to find corresponding events between spacecraft, but only set on discontinuities. 
        The Chi^2 minimum value relates to a given spacecraft and the first spacecraft in the craft list (Default = False).
    use_dtw: boolean, optional
        Dynamic time warping to find the best off set time (Default= False).
    plot      : boolean, optional
        Plot the Chi^2 minimum values as a function of time for a given spacecraft with respect to the reference spacecraft
        (Default = True). You should keep this parameter on because the output html file references this created file.
    refine:  boolean, optional
        Refine Chi^2 minimization by loop with smaller ranges,windows around previous minimum at the highest possible
        observational cadence (Default = True).
    verbose: boolean, optional
        Print the process time for an event (Default = True).
    nproc  : integer, optional
        Number of processors to use in Chi^2 minimization (Default = 1). For 1 event I see a 30% time decrease by using
        nproc=8 over nproc=1.
    p_val : float, optional
        Probability value (0<p_val<1.0) used to define an event (Default = 0.999). Do not set value below 0.90 for two reasons.
        One that creates a lot of "bad events" (i.e. events not seen in all four spacecraft or use instrument spikes.) two 
        currently the code has a cut of 0.90 to define an event start.
    ref_chi_t : pandas time delta object, optional
        Size of window around an event (+/-) to find the Chi^2 minimum for a refined window (Default = pd.to_timedelta('30 minutes')).
        During refinement the window will run once and shrink two times by 1+loop (Default = 30,15,10).
    rgh_chi_t : pandas time delta object, optional
        Size of window around an event (+/-) to find the Chi^2 minimum for a rough window (Default = pd.to_timedelta('90 minutes')).
        During the rough minimization the window will run once and shrink once by 1+loop (Default = 90,45).
    arch: string, optional
        The archive location for the text file to read in (Default = '../cdf/cdftotxt/')
    mag_fmt: string, optional
        The file format for the magnetic field observations (Default = '{0}_mag_2015_2017_formatted.txt',
        where 0 is the formatted k).
    pls_fmt: string, optional
        The file format for the plasma observations (Default = '{0}_pls_2015_2017_formatted.txt',
        where 0 is the formatted k).
    orb_fmt: string, optional
        The file format for the orbital data (Default = '{0}_orb_2015_2017_formatted.txt',
        where 0 is the formatted k).
    start_t: string, optional
        Date in YYYY/MM/DD format to start looking for events (Default = '2016/06/04')
    start_t: string, optional
        Date in YYYY/MM/DD format to stop looking for events (inclusive, Default = '2017/07/31')
    center = boolean, optional
        Whether the analyzed point to be center focused (center = True) or right focus (Default = False).
        Using a right focused model I find better agreement with events found by eye.
    Returns
    -------
    '''


model_20161221_0843.py
---------------------
Old, DTW model of a single event.

model_time_range.py
-------------------
Find continuos DTW solution over a given time range.

        """
        Class to get planar DTW solutions for L1 spacecraft.      
 
        Parameters:
        ----------
        start_t: string
            Any string format recongized by pd.to_datetime indicating when to start looking for events
        end_t: string
            Any string format recongized by pd.to_datetime indicating when to stop looking for events
        center: boolean, optional
            Whether to use the center pixel for a running mean (Default = True). Otherwise running mean
            is set by preceding pixels
        events: int,optional
            Number of Events to find planar solution for in given time period (Default = 1)
        par: string or list, optional
            Parameter to use when matching via DTW (Default = None). The default solution is to use 
            flow speed for SOHO CELIAS and maximum difference in a 3 minute window for magnetic 
            field component for every other spacecraft.
        justparm: boolean, optional
            Just do DTW solution but do not create animation of solution as a funciton of time
            (Default = True)
        nproc: integer, optional
            Number of processors to use for matching (Default = 1). Currently, there is no reason
            to change this value, but is a place holder incase someday it becomes useful
        earth_craft: list, optional 
            Show Themis/Artemis space craft and the best solutions (Default = None). Can be 
            any combinateion of ['THEMIS_B','THEMIS_C','THEMIS_A'] 
        penalty: boolean, optional
            Include a penalty in the DTW solution for compression of time (Default = True)
        pad_earth: pandas time delta object, optional
            Time offset to apply when reading in spacecraft data near earth (Default = pd.to_timedelta('1 hour'))
            
        Example: 
        ----------
        import model_time_range as mtr
        plane = mtr.dtw_plane('2016/07/19 21:00:00','2016/07/20 01:00:00',earth_craft=['THEMIS_B'],penalty=False)
        plane.init_read()
        plane.main()
       
        """

plot_soho.py
-----------
Interactively plot SOHO observations.

plot_space_craft_distributions.py
---------------
Old

shock_times.txt
-------------
List of shock times

unit_test_model_time.py
----------------------
Test whether the planar solutions in model_time_range are accurate.
Usage:

python unit_test_model_time.py