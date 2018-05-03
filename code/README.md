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