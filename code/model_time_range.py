import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from spacepy import pycdf
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import os
import threading
import sys
import time
import mlpy #for dynamic time warping 
from dtaidistance import dtw #try new dynamic time warping function that creates penalty for compression
import load_cdf_files as lcf #reading cdfs to pandas arrays


from scipy.stats.mstats import theilslopes
import scipy.optimize


#Defination of a plane
def plane_func(a,b,c,d,x,y,z):
    return a*x+b*y+c*z+d


def solve_plane(p,t):
    """
    Velocity plane for given time and position of spacecraft
    
    Parameters:
    ---------
    p: np.array or np.matrix
        Position vectors in x,y,z for three spacecraft with respect to wind
        The First row is the X,Y,Z values for spacecraft 1
        The Second row is the X,Y,Z values for spacecraft 2
        The Thrid row is the X,Y,Z values for spacecraft 3
    t: np.array or np.matrix
        Time offset array from Wind for three spacecraft
    """
    vna = np.linalg.solve(p,t) #solve for the velocity vectors normal
    vn  = vna/np.linalg.norm(vna)
    vm  = 1./np.linalg.norm(vna) #get velocity magnitude
    return vna,vn,vm

def solve_coeff(pi,vn):
    """
    Plane coefficients for given time and position of spacecraft
    
    Parameters:
    ----------
    pi: np.array or np.matrix
        Position of earth spacecraft in GSE corrected for time offsets
        The First row is the X,Y,Z values for spacecraft 1
        The Second row is the X,Y,Z values for spacecraft 2
        The Thrid row is the X,Y,Z values for spacecraft 3
    vn: np.array or np.matrix
        Normal vector to planar front
  
    Returns:
    ----------
    a,b,c,d: float
        Solution for a plane at time t where plane has the solution 0=a*x+b*y+c*z+d
        
    """

    #solve (a*x0+b*y0+c*z0+d)/||v|| = 0
    #where ||v|| = sqrt(a^2+b^2+c^2)
    #and distance from the plane to the origin is  dis = d/||v||
    

    #get the magnitude of the normal vector on the plane from the origin
    pm = float(np.linalg.norm(pi))

    #Use definition parameter 
    coeff = np.squeeze(np.asarray(vn*pm)) #.reshape(-1)


    #store coefficients
    a = float(coeff[0])
    b = float(coeff[1])
    c = float(coeff[2])
    d = -float(np.matrix(coeff).dot(pi))
    
    return [a,b,c,d]

#Function to read in spacecraft
def read_in(k,p_var='predict_shock_500',arch='../cdf/cdftotxt/',
            mag_fmt='{0}_mag_2015_2017_formatted.txt',pls_fmt='{0}_pls_2015_2017_formatted.txt',
            orb_fmt='{0}_orb_2015_2017_formatted.txt',
            start_t='2016/12/01',end_t='2017/09/24',center=False):
    """
    A function to read in text files for a given spacecraft

    Parameters
    ----------
    k: string
        Then name of a spacecraft so to format for file read in
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
    center = boolean, optional
        Whether the analyzed point to be center focused (center = True) or right focus (Default = False).
        Center focus gives you a better localized point, however, the model is trained with a right focus
        in order to reject spikes and increase S/N.
    start_t: string, optional
        Date in YYYY/MM/DD format to start looking for events (Default = '2016/06/04')
    end_t: string, optional
        Date in YYYY/MM/DD format to stop looking for events (inclusive, Default = '2017/07/31')

    Returns
    -------
    plsm: Pandas DataFrame
        A pandas dataframe with probability values and combined mag and plasma observations.
    
    """
    #Read in plasma and magnetic field data from full res
    if k.lower() == 'soho':
        pls = pd.read_csv(arch+pls_fmt.format(k.lower()),delim_whitespace=True)
    #Change to function that reads cdf files for less data intense loads 2018/05/17 J. Prchlik
    else:
        outp = lcf.main(pd.to_datetime(start_t),pd.to_datetime(end_t),scrf=[k.lower()],pls=True,mag=True,orb=True)
        pls = outp[k.lower()]['pls']
    Re = 6371.0 # Earth radius in km

    #no magnetic field data from SOHO
    if k.lower() != 'soho':
        #Change to function that reads cdf files for less data intense loads 2018/05/17 J. Prchlik
        #Add data quality cut 2018/05/18 J. Prchlik
        if k.lower == 'dscovr':
            good_mag = (outp[k.lower()]['mag'].DQF == 0)
            mag = outp[k.lower()]['mag'][good_mag]
        else:
            mag = outp[k.lower()]['mag']


        orb = outp[k.lower()]['orb']

        #create datetime objects from time
        pls['time_dt_pls'] = pd.to_datetime(pls['Time'])
        mag['time_dt_mag'] = pd.to_datetime(mag['Time'])
        orb['time_dt_orb'] = pd.to_datetime(orb['Time'])

        #setup index
        pls.set_index(pls.time_dt_pls,inplace=True)
        mag.set_index(mag.time_dt_mag,inplace=True)
        orb.set_index(orb.time_dt_orb,inplace=True)

        #multiply each component by Earth Radius for Themis observations
        if 'themis' in k.lower():
            orb.loc[:,'GSEx'] *= Re
            orb.loc[:,'GSEy'] *= Re
            orb.loc[:,'GSEz'] *= Re
            #Convert from GSM to GSE 2018/04/25 J. Prchlik
            mag.loc[:,'By'] *= -1
            mag.loc[:,'Bz'] *= -1


        #cut for testing reasons
        pls = pls[start_t:end_t]
        mag = mag[start_t:end_t]
        orb = orb[start_t:end_t]

        #pls = pls['2016/07/18':'2016/07/21']
        #mag = mag['2016/07/18':'2016/07/21']
        #pls = pls['2017/01/25':'2017/01/27']
        #mag = mag['2017/01/25':'2017/01/27']

        #join magnetic field and plasma dataframes
        com_df  = pd.merge(mag,pls,how='outer',left_index=True,right_index=True,suffixes=('_mag','_pls'),sort=True)

        #make sure data columns are numeric
        cols = ['SPEED','Np','Vth','Bx','By','Bz']
        com_df[cols] = com_df[cols].apply(pd.to_numeric, errors='coerce')

        #add Time string
        com_df['Time'] = com_df.index.to_datetime().strftime('%Y/%m/%dT%H:%M:%S')

        plsm = com_df
        #replace NaN with previously measured value
        #com_df.fillna(method='bfill',inplace=True)

        #add orbital data
        plsm  = pd.merge(plsm,orb,how='outer',left_index=True,right_index=True,suffixes=('','_orb'),sort=True)
        #make sure data columns are numeric
        cols = ['SPEED','Np','Vth','Bx','By','Bz','GSEx','GSEy','GSEz']
        plsm[cols] = plsm[cols].apply(pd.to_numeric, errors='coerce')

        #add Time string
        plsm['Time'] = plsm.index.to_datetime().strftime('%Y/%m/%dT%H:%M:%S')

        #fill undersampled orbit
        for cor in ['x','y','z']: plsm['GSE'+cor].interpolate(inplace=True)

    else:
        plsm = pls   
        #work around for no Mag data in SOHO
        pls.loc[:,['Bx','By','Bz']] = 0.0
        pls['time_dt_pls'] = pd.to_datetime(pls['Time'])
        pls['time_dt_mag'] = pd.to_datetime(pls['Time'])
        pls.set_index(pls.time_dt_pls,inplace=True)
        plsm = pls[start_t:end_t]
        plsm.loc[:,['Bx','By','Bz']] = -9999.0

        Re = 6371.0 # Earth radius

        #multiply each component by Earth Radius
        plsm.loc[:,'X'] *= Re
        plsm.loc[:,'Y'] *= Re
        plsm.loc[:,'Z'] *= Re

        #chane column name from X, Y, Z to GSEx, GSEy, GSEz 
        plsm.rename(columns={'X':'GSEx', 'Y':'GSEy', 'Z':'GSEz'},inplace=True)

    #force index to sort
    plsm.sort_index(inplace=True)
    #for rekeying later
    plsm['craft'] = k

    return plsm




class dtw_plane:


    def __init__(self,start_t,end_t,center=True,events=1,par=None,justparm=True,nproc=1,earth_craft=None,penalty=True,pad_earth=pd.to_timedelta('1 hour'),speed_pen=10.,mag_pen=0.2):
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
        speed_pen: float
            Penatly in km/s for squashing speed time in DTW (Default = 10.). Only works if penalty is set to True
        mag_pen: float
            Penatly in nT for squashing magnetic field time in DTW (Default = 0.2). Only works if penalty is set to True.

        Example: 
        ----------
        import model_time_range as mtr
        plane = mtr.dtw_plane('2016/07/19 21:00:00','2016/07/20 01:00:00',earth_craft=['THEMIS_B'],penalty=False)
        plane.init_read()
        plane.dtw()
       

        """
        self.start_t = start_t
        self.end_t = end_t
        self.center = center
        self.par = par
        self.justparm = justparm
        self.nproc = nproc
        self.Re = 6371.0 #km
        self.events = events
        self.earth_craft = earth_craft
        self.penalty = penalty
        self.pad_earth = pad_earth

        self.first = True


        #store penanalties
        self.speed_pen = speed_pen
        self.mag_pen = mag_pen


        #set use to use all spacecraft
        self.craft = ['Wind','DSCOVR','ACE','SOHO','THEMIS_A','THEMIS_B','THEMIS_C']
        self.col   = ['blue','black','red','teal','purple','orange','cyan']
        self.mar   = ['D','o','s','<','>','^','8']
        self.marker = {}
        self.color  = {}
        self.trainer = 'Wind'

        #create dictionaries for labels
        for j,i in enumerate(self.craft):
            self.marker[i] = self.mar[j]
            self.color[i]  = self.col[j]



        #reset craft variable and add earth craft as requested
        self.craft = ['Wind','DSCOVR','ACE','SOHO']
      
        
    



    def init_read(self):
        """
        Reads in text files containing information on solar wind parameters measured at different space craft

        Parameters
        ----------
        self: class
            Variables contained in self variable
        """
        #Parameters for file read in and parsing
        #Could be an issue with downwind THEMIS craft 2018/04/25 J. Prchlik
        par_read_in = partial(read_in,start_t=self.start_t,end_t=self.end_t,center=self.center)
        #read in and format spacecraft in parallel
        #Switched to single loop solution 2018/03/24 J. Prchlik 
        if self.first: #only do read in on the first pass
            if self.nproc > 1.5:
                pool = Pool(processes=len(self.craft))
                outp = pool.map(par_read_in,self.craft)
                pool.terminate()
                pool.close()
                pool.join()

                self.plsm = {}
                #create global plasma key
                for i in outp:
                    self.plsm[i.craft.values[0]] = i
            else:
                self.plsm = {}
                #create global plasma key
                for i in self.craft:
                    self.plsm[i] = par_read_in(i)
        
            #set readin to first attempt to false
            #prevent multiple readin of big files
            self.first = False


            #do the same for the Earth spacecraft 
            if self.earth_craft is not None:  
                for i in self.earth_craft: self.craft.append(i)
 
                 
                #Add an hour to the data to approximate time delay
                self.earth_start = str(pd.to_datetime(self.start_t)+self.pad_earth)
                self.earth_end = str(pd.to_datetime(self.end_t)+self.pad_earth)
                par_read_in_e = partial(read_in,start_t=self.earth_start,end_t=self.earth_end,center=self.center)

                if self.nproc > 1.5:
                    pool = Pool(processes=len(self.earth_craft))
                    outp = pool.map(par_read_in_e,self.earth_craft)
                    pool.terminate()
                    pool.close()
                    pool.join()

                    #create global plasma key
                    for i in outp:
                        self.plsm[i.craft.values[0]] = i
                else:
                    #create global plasma key
                    for i in self.earth_craft:
                        self.plsm[i] = par_read_in_e(i)
        
            #set readin to first attempt to false
            #prevent multiple readin of big files

    def dtw(self):
        """
        Finds planar solution to 4 L1 spacecraft
     
        Parameters
        ----------
 
        self: Class

 
        """
        #Creating modular solution for DTW 2018/03/21 J. Prchlik
        ##set the Start and end time
        #start_t = "2016/12/21 07:00:00"
        #end_t = "2016/12/21 13:00:00"
        #center = True
        #reset variables to local variables
        Re = self.Re # Earth radius
        start_t  = self.start_t 
        end_t    = self.end_t   
        center   = self.center  
        par      = self.par     
        justparm = self.justparm
        marker   = self.marker
        color    = self.color
        
        #set use to use all spacecraft
        craft = self.craft #['Wind','DSCOVR','ACE','SOHO']
        col   = self.col   #['blue','black','red','teal']
        mar   = self.mar   #['D','o','s','<']
        trainer = self.trainer
        
        #range to find the best maximum value
        maxrang = pd.to_timedelta('3 minutes')
        
        
        #create new plasma dictory which is a subset of the entire file readin
        plsm = {}
        for i in craft:
             plsm[i] = self.plsm[i] #[start_t:end_t] Cutting not required because already using a small sample 2018/05/03 J. Prchlik
             #remove duplicates
             plsm[i] = plsm[i][~plsm[i].index.duplicated(keep='first')]

        #get all values at full resolution for dynamic time warping
        t_mat  = plsm[trainer] #.loc[trainer_t-t_rgh_wid:trainer_t+t_rgh_wid]
  
        #add trainer matrix to self
        self.t_mat = t_mat

        #Find points with the largest speed differences in wind
        top_vs = (t_mat.SPEED.dropna().diff().abs()/t_mat.SPEED.dropna()).nlargest(self.events)
        
        
        #sort by time for event number
        top_vs.sort_index(inplace=True)
        
        #add to self
        self.top_vs = top_vs

        #plot with the best timing solution
        self.fig, self.fax = plt.subplots(ncols=2,nrows=3,sharex=True,figsize=(18,18))
        fig, fax = self.fig,self.fax

       
        #set range to include all top events (prevents window too large error
        self.pad = pd.to_timedelta('30 minutes')
        pad = self.pad
        fax[0,0].set_xlim([top_vs.index.min()-pad,top_vs.index.max()+pad])
        
        #loop over all other craft
        for k in craft[1:]:
            print('###########################################')
            print(k)
            p_mat  = plsm[k] #.loc[i_min-t_rgh_wid:i_min+t_rgh_wid]
        
            #use speed for rough esimation if possible
            if  ((k.lower() == 'soho') ): par = ['SPEED']
            elif (((par is None) | (isinstance(par,float))) & (k.lower() != 'soho')): par = ['Bx','By','Bz']
            elif isinstance(par,str): par = [par]
            else: par = par

        
            #sometimes different componets give better chi^2 values therefore reject the worst when more than 1 parameter
            #Try using the parameter with the largest difference  in B values preceding and including the event (2017/12/11 J. Prchlik)
            if len(par) > 1:
               check_min,check_max = top_vs.index[0]-maxrang,top_vs.index[0]+maxrang
               par_chi = np.array([(t_mat.loc[check_min:check_max,par_i].max()-t_mat.loc[check_min:check_max,par_i].min()).max() for par_i in par])
               use_par, = np.where(par_chi == np.max(par_chi))
               par      = list(np.array(par)[use_par])
        
            #get the median slope and offset
            #J. Prchlik (2017/11/20)
            #Dont use interpolated time for solving dynamic time warp (J. Prchlik 2017/12/15)
            #only try SPEED corrections for SOHO observations
            #Only apply speed correction after 1 iteration (J. Prchlik 2017/12/18)
            if (('themis' in k.lower())):
                try:
                    #create copy of p_mat
                    c_mat = p_mat.copy()
                    #resample the matching (nontrained spacecraft to the trained spacecraft's timegrid to correct offset (2017/12/15 J. Prchlik)
                    c_mat = c_mat.reindex(t_mat.index,method='nearest').interpolate('time')
         
                    #only comoare no NaN values
                    good, = np.where(((np.isfinite(t_mat.SPEED.values)) & (np.isfinite(c_mat.SPEED.values))))
         
                    #if few points for comparison only used baseline offset
                    if ((good.size < 1E36) & (par[0] == 'SPEED')):
                        med_m,med_i = 1.0,0.0
                        off_speed = p_mat.SPEED.median()-t_mat.SPEED.median()
                        p_mat.SPEED = p_mat.SPEED-off_speed
                        if med_m > 0: p_mat.SPEED = p_mat.SPEED*med_m+med_i
                    else:
                        off_speed = p_mat.SPEED.nsmallest(100).median()-t_mat.SPEED.nsmallest(20).median()
                        p_mat.SPEED = p_mat.SPEED-off_speed
                    #only apply slope if greater than 0
                except IndexError:
                #get median offset to apply to match spacecraft
                    off_speed = p_mat.SPEED.nsmallest(100).median()-t_mat.SPEED.nsmallest(20).median()
                    p_mat.SPEED = p_mat.SPEED-off_speed
         
         
         
            #get dynamic time warping value   
            print('WARPING TIME')
            #use dtw solution that allows penalty for time compression
            if self.penalty:
                if 'SPEED' in par:
                    penalty = self.speed_pen
                elif any('B' in s for s in par):
                    penalty = self.mag_pen
 
                print('Penalty = {0:4.3f}'.format(penalty))
                path = dtw.warping_path(t_mat[par[0]].ffill().bfill().values,
                                        p_mat[par[0]].ffill().bfill().values,
                                        penalty=penalty)
                #reformat in old format 2018/04/20 J. Prchlik
                path = np.array(path).T
            #Otherwise you quick DTW solution
            else:
                dist, cost, path = mlpy.dtw_std(t_mat[par[0]].ffill().bfill().values,p_mat[par[0]].ffill().bfill().values,dist_only=False)
            print('STOP WARPING TIME')
        
            #get full offsets for dynamic time warping
            off_sol = (p_mat.iloc[path[1],:].index - t_mat.iloc[path[0],:].index)
            print('REINDEXED')
        
            #get a region around one of the best fit times
            b_mat = p_mat.copy()
        
            #update the time index of the match array for comparision with training spacecraft (i=training spacecraft time)
            b_mat = b_mat.reindex(b_mat.iloc[path[1],:].index) #.interpolate('time')
            b_mat.index = b_mat.index-off_sol
            b_mat['offsets'] = off_sol
        
            #Add offset data frame to plasma diction
            plsm[k+'_offset'] = b_mat
            #plot plasma parameters
            #fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k.upper())         
            if len(b_mat[b_mat['Np'   ] > -9990.0]) > 0:
                fax[0,0].scatter(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,marker=marker[k],color=color[k],label=k)
                fax[0,0].plot(b_mat[b_mat['Np'   ] > -9990.0].index,b_mat[b_mat['Np'   ] > -9990.0].Np   ,color=color[k],linewidth=2,label='')
        
            if len(b_mat[b_mat['Vth'  ] > -9990.0]) > 0:
                fax[1,0].scatter(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[k],color=color[k],label=k)
                fax[1,0].plot(b_mat[b_mat['Vth'  ] > -9990.0].index,b_mat[b_mat['Vth'  ] > -9990.0].Vth  ,color=color[k],linewidth=2,label='')
        
            if len(b_mat[b_mat['SPEED'] > -9990.0]) > 0:
                fax[2,0].scatter(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,marker=marker[k],color=color[k])
                fax[2,0].plot(b_mat[b_mat['SPEED'] > -9990.0].index,b_mat[b_mat['SPEED'] > -9990.0].SPEED,color=color[k],linewidth=2)
        
        
            #plot mag. parameters
            if k.lower() != 'soho':
                if len(b_mat[b_mat['Bx']    > -9990.0]) > 0:
                    fax[0,1].scatter(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,marker=marker[k],color=color[k])
                    fax[0,1].plot(b_mat[b_mat['Bx']    > -9990.0].index,b_mat[b_mat['Bx']    > -9990.0].Bx,color=color[k],linewidth=2)
        
                if len(b_mat[b_mat['By']    > -9990.0]) > 0:
                    fax[1,1].scatter(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,marker=marker[k],color=color[k])
                    fax[1,1].plot(b_mat[b_mat['By']    > -9990.0].index,b_mat[b_mat['By']    > -9990.0].By,color=color[k],linewidth=2)
        
                if len(b_mat[b_mat['Bz']    > -9990.0]) > 0:
                    fax[2,1].scatter(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,marker=marker[k],color=color[k])
                    fax[2,1].plot(b_mat[b_mat['Bz']    > -9990.0].index,b_mat[b_mat['Bz']    > -9990.0].Bz,color=color[k],linewidth=2)
        
        
            print('###########################################')
        
        
        #set 0 offsets for training spacecraft
        t_mat['offsets'] = pd.to_timedelta(0) 
        plsm[trainer+'_offset'] = t_mat
        
        
        #plot plasma parameters for Wind
        if len(t_mat[t_mat['Np'   ] > -9990.0]) > 0:
            fax[0,0].scatter(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,marker=marker[trainer],color=color[trainer],label=trainer.upper())
            fax[0,0].plot(t_mat[t_mat['Np'   ] > -9990.0].index,t_mat[t_mat['Np'   ] > -9990.0].Np   ,color=color[trainer],linewidth=2,label='')
        
        if len(t_mat[t_mat['Vth'  ] > -9990.0]) > 0:
            fax[1,0].scatter(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,marker=marker[trainer],color=color[trainer],label=trainer)
            fax[1,0].plot(t_mat[t_mat['Vth'  ] > -9990.0].index,t_mat[t_mat['Vth'  ] > -9990.0].Vth  ,color=color[trainer],linewidth=2,label='')
        
        if len(t_mat[t_mat['SPEED'] > -9990.0]) > 0:
            fax[2,0].scatter(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,marker=marker[trainer],color=color[trainer])
            fax[2,0].plot(t_mat[t_mat['SPEED'] > -9990.0].index,t_mat[t_mat['SPEED'] > -9990.0].SPEED,color=color[trainer],linewidth=2)
        
        
        #plot mag. parameters
        if len(t_mat[t_mat['Bx']    > -9990.0]) > 0:
            fax[0,1].scatter(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,marker=marker[trainer],color=color[trainer])
            fax[0,1].plot(t_mat[t_mat['Bx'   ] > -9990.0].index,t_mat[t_mat['Bx']    > -9990.0].Bx,color=color[trainer],linewidth=2)
        
        if len(t_mat[t_mat['By']    > -9990.0]) > 0:
            fax[1,1].scatter(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,marker=marker[trainer],color=color[trainer])
            fax[1,1].plot(t_mat[t_mat['By'   ] > -9990.0].index,t_mat[t_mat['By']    > -9990.0].By,color=color[trainer],linewidth=2)
        
        if len(t_mat[t_mat['Bz']    > -9990.0]) > 0:
            fax[2,1].scatter(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,marker=marker[trainer],color=color[trainer])
            fax[2,1].plot(t_mat[t_mat['Bz'   ] > -9990.0].index,t_mat[t_mat['Bz']    > -9990.0].Bz,color=color[trainer],linewidth=2)
        
        
        fancy_plot(fax[0,0])
        fancy_plot(fax[1,0])
        fancy_plot(fax[2,0])
        fancy_plot(fax[0,1])
        fancy_plot(fax[1,1])
        fancy_plot(fax[2,1])
        #i = pd.to_datetime("2016/12/21 08:43:12") 
        fax[0,0].set_xlim([start_t,end_t])
        
        fax[0,0].set_ylabel('Np [cm$^{-3}$]',fontsize=20)
        fax[1,0].set_ylabel('Th. Speed [km/s]',fontsize=20)
        fax[2,0].set_ylabel('Flow Speed [km/s]',fontsize=20)
        fax[2,0].set_xlabel('Time [UTC]',fontsize=20)
        
        fax[0,1].set_ylabel('Bx [nT]',fontsize=20)
        fax[1,1].set_ylabel('By [nT]',fontsize=20)
        fax[2,1].set_ylabel('Bz [nT]',fontsize=20)
        fax[2,1].set_xlabel('Time [UTC]',fontsize=20)
        
        fax[1,0].set_ylim([0.,100.])
        
        
        #turn into data frame 
        frm_vs = pd.DataFrame(top_vs)
        #add columns
        col_add = ['X','Y','Z','Vx','Vy','Vz']
        for i in col_add: frm_vs[i] = -9999.9



        #Updated self plasma dictionary
        self.plsm = plsm
        self.fig, self.fax = fig,fax
        
        #Do not need this 2018/03/21 J. Prchlik
        ####Use wind CDF to get velocity comps
        ####cdf = pycdf.CDF('/Volumes/Pegasus/jprchlik/dscovr/solar_wind_events/cdf/wind/plsm/wi_h1_swe_20161221_v01.cdf')
        ####
        ####wind_vx = cdf['Proton_VX_nonlin'][...]
        ####wind_vy = cdf['Proton_VY_nonlin'][...]
        ####wind_vz = cdf['Proton_VZ_nonlin'][...]
        ####wind_t0 = cdf['Epoch'][...]
        ####
        ####cdf.close()
        ####
        #####create pandas dataframe with wind components
        ####wind_v = pd.DataFrame(np.array([wind_t0,wind_vx,wind_vy,wind_vz]).T,columns=['time_dt','Vx','Vy','Vz'])
        ####wind_v.set_index(wind_v.time_dt,inplace=True)
        #big list of velocities
        #big_lis = []

    def pred_earth(self):
        """
        Create prediction for Earth and create corresponding plots

        """

        #Use common names for self variables
        t_mat = self.t_mat
        plsm  = self.plsm
        Re = self.Re # Earth radius
        start_t  = self.start_t 
        end_t    = self.end_t   
        center   = self.center  
        par      = self.par     
        justparm = self.justparm
        marker   = self.marker
        color    = self.color
        pad = self.pad
        fig, fax = self.fig,self.fax
        
        #set use to use all spacecraft
        craft = self.craft #['Wind','DSCOVR','ACE','SOHO']
        col   = self.col   #['blue','black','red','teal']
        mar   = self.mar   #['D','o','s','<']
        trainer = self.trainer
        
        #range to find the best maximum value
        maxrang = pd.to_timedelta('3 minutes')

        #Find points with the largest speed differences in wind
        #Allow dyanic allocation of top events 2018/05/17 J. Prchlik
        top_vs = (plsm[trainer].SPEED.dropna().diff().abs()/plsm[trainer].SPEED.dropna()).nlargest(self.events)
        
        
        #sort by time for event number
        top_vs.sort_index(inplace=True)



        


        #Add plot for prediction on THEMIS
        fig_th,ax_th = plt.subplots()
        #Add plot with just the THEMIS plasma data
        for esp in self.earth_craft:
            slicer = np.isfinite(plsm[esp].SPEED)
            ax_th.plot(plsm[esp].loc[slicer,:].index,pd.rolling_mean(plsm[esp].loc[slicer,:].SPEED,25),color=color[esp],label=esp.upper(),zorder=100,linewidth=2)

        ax_th.set_xlim([pd.to_datetime(self.start_t)-pad,pd.to_datetime(self.end_t)+pad])
        ax_th.set_xlabel('Time [UTC]')
        ax_th.set_ylabel('Flow Speed [km/s]')
        fancy_plot(ax_th)


        #create dictionary of values for each event 2018/04/24 J. Prchlik
        self.event_dict = {}

        #List of time and Speed value of events J. Prchlik
        event_plot = []
        
        #Plot the top shock values
        #fax[2,0].scatter(t_mat.loc[top_vs.index,:].index,t_mat.loc[top_vs.index,:].SPEED,color='purple',marker='X',s=150)
        for j,i in enumerate(top_vs.index):
        #try producing continous plot 2018/05/17 J. Prchlik
        #This method did not work
        #for j,i in enumerate(t_mat.index):
            yval = t_mat.loc[i,:].SPEED
            yvalb = 0.
            xval = mdates.date2num(i)

            #try producing continous plot 2018/05/17 J. Prchlik
            fax[2,0].annotate('Event {0:1d}'.format(j+1),xy=(xval,yval),xytext=(xval,yval+50.),
                              arrowprops=dict(facecolor='purple',shrink=0.005))
            #fax[2,1].annotate('Event {0:1d}'.format(j+1),xy=(xval,yvalb),xytext=(xval,yvalb+2.),
            #                  arrowprops=dict(facecolor='purple',shrink=0.005))


            #computer surface for events
            #tvals = -np.array([np.mean(plsm[c+'_offset'].loc[i,'offsets']).total_seconds() for c in craft])
            #xvals = np.array([np.mean(plsm[c].loc[i,'GSEx']) for c in craft])
            #yvals = np.array([np.mean(plsm[c].loc[i,'GSEy']) for c in craft])
            #zvals = np.array([np.mean(plsm[c].loc[i,'GSEz']) for c in craft])
            #Switched to one loop 2018/03/07
            tvals = [] #-np.array([np.mean(plsm[c+'_offset'].loc[i,'offsets']).total_seconds() for c in craft])
            xvals = [] #np.array([np.mean(plsm[c].loc[i,'GSEx']) for c in craft])
            yvals = [] #np.array([np.mean(plsm[c].loc[i,'GSEy']) for c in craft])
            zvals = [] #np.array([np.mean(plsm[c].loc[i,'GSEz']) for c in craft])
         
            #create master event dictionary for given event to store parameters
            cur = 'event_{0:1d}'.format(j+1)
            self.event_dict[cur] = {}
           
        
            #loop over all craft and populate time and position arrays
            for c in craft:
                #append craft values onto time and position arrays
                #changed to min values 2018/03/12 J. Prchlik
                try:
                    itval = plsm[c+'_offset'].loc[i,:].offsets
                #Fix THEMIS having out of range index
                except KeyError:
                    check = plsm[c+'_offset'].GSEx.dropna().index.get_loc(i,method='nearest')
                    it = plsm[c+'_offset'].GSEx.dropna().index[check]
                    itval = plsm[c+'_offset'].loc[it,:].offsets
                    
                if isinstance(itval,pd._libs.tslib.Timedelta):
                    off_cor = itval.total_seconds()
                    tvals.append(itval.total_seconds())
                elif isinstance(itval,pd.Series):
                    tvals.append(min(itval,key=abs).total_seconds())
                    off_cor = min(itval,key=abs).total_seconds()

                #Get closest index value location
                #Update with time offset implimented
                ii = plsm[c].GSEx.dropna().index.get_loc(i+pd.to_timedelta(off_cor,unit='s'),method='nearest')
                #convert index location back to time index
                it = plsm[c].GSEx.dropna().index[ii]

                #Use offset pandas DF position 2018/04/25 J. Prchlik
                xvals.append(np.mean(plsm[c].iloc[ii,:].GSEx))
                yvals.append(np.mean(plsm[c].iloc[ii,:].GSEy))
                zvals.append(np.mean(plsm[c].iloc[ii,:].GSEz))
        
            #Covert arrays into numpy arrays and flip sign of offset
            self.event_dict[cur]['tvals'] = np.array(tvals)
            self.event_dict[cur]['xvals'] = np.array(xvals) 
            self.event_dict[cur]['yvals'] = np.array(yvals) 
            self.event_dict[cur]['zvals'] = np.array(zvals) 
            #Print position values and time values
        
            #get the velocity components with respect to the shock front at wind
            #i_val = wind_v.index.get_loc(i,method='nearest')
            #vx = wind_v.iloc[i_val].Vx
            #vy = wind_v.iloc[i_val].Vy
            #vz = wind_v.iloc[i_val].Vz
            #use positions and vectors to get a solution for plane velocity
            pm  = np.matrix([xvals[1:4]-xvals[0],yvals[1:4]-yvals[0],zvals[1:4]-zvals[0]]).T #coordinate of craft 1 in top row
            tm  = np.matrix(tvals[1:4]).T # 1x3 matrix of time (wind-spacecraft)
            vna,vn,vm = solve_plane(pm,tm)
            #vna = np.linalg.solve(pm,tm) #solve for the velocity vectors normal
            #vn  = vna/np.linalg.norm(vna)
            #vm  = 1./np.linalg.norm(vna) #get velocity magnitude
            
            #store vx,vy,vz values
            self.event_dict[cur]['vx'],self.event_dict[cur]['vy'],self.event_dict[cur]['vz'] = vm*np.array(vn).ravel()
            #store normal vector 2018/04/24 J. prchlik
            self.event_dict[cur]['vn'] = vn
            self.event_dict[cur]['vm'] = vm
        
            #get the 4 point location of the front when at wind
            #p_x(t0)1 = p_x(t1)-V_x*dt where dt = t1-t0  
            #solving exactly
            #use the velocity matrix solution to get the solution for the plane analytically
            #2018/03/15 J. Prchlik
            #px = -vx*tvals+xvals
            #py = -vy*tvals+yvals
            #pz = -vz*tvals+zvals
            self.event_dict[cur]['wind_px'] = xvals[0]
            self.event_dict[cur]['wind_py'] = yvals[0]
            self.event_dict[cur]['wind_pz'] = zvals[0]

            #Wind position to determine starting point  2018/05/24 J. Prchlik
            px = xvals[0]
            py = yvals[0]
            pz = zvals[0]

            for esp in self.earth_craft:
                ################################################################
                #Get THEMIS B location and compare arrival times
                ################################################################
                #Get closest index value location
                #Fix THEMIS having out of range index
                try:
                    itval = plsm[esp+'_offset'].loc[i,:].offsets
                    #Get time of observation in THEMIS B
                    itind = pd.to_datetime(plsm[esp+'_offset'].loc[i,'Time'])
                    it = i
                #Fix THEMIS having out of range index
                except KeyError:
                    check = plsm[esp+'_offset'].GSEx.dropna().index.get_loc(i,method='nearest')
                    it = plsm[esp+'_offset'].GSEx.dropna().index[check]
                    itval = plsm[esp+'_offset'].loc[it,:].offsets
                    #Get time of observation in THEMIS B
                    itind = pd.to_datetime(plsm[esp+'_offset'].loc[it,'Time'])

                #Get first match if DTW produces more than one
                if isinstance(itind,pd.Series): itind = itind.dropna()[1]
                if isinstance(itval,pd._libs.tslib.Timedelta):
                    atval = itval.total_seconds()
                elif isinstance(itval,pd.Series):
                    atval = np.mean(itval).total_seconds() #,key=abs).total_seconds()

                #Store THEMIS position
                axval = np.mean(plsm[esp+'_offset'].loc[it,'GSEx'])
                ayval = np.mean(plsm[esp+'_offset'].loc[it,'GSEy'])
                azval = np.mean(plsm[esp+'_offset'].loc[it,'GSEz'])

                ################################################################
                ################################################################

                #parameters to add
                #Switched to dictionary 2018/04/24 J. Prchlik
                #add_lis = [vx,vy,vz,tvals,vm,vn,px,py,pz]
                #big_lis.append(add_lis)
                #Wind Themis B distance difference from plane at wind
                themis_d = float(vn.T.dot((np.matrix([axval,ayval,azval])-np.matrix([px,py,pz])).T))
                print(vm)
                print(themis_d)
                themis_dt = float(themis_d)/vm
                themis_pr = i+pd.to_timedelta(themis_dt,unit='s')

                #Try to print prediction, but if it fails just move on 2018/05/17 J. Prchlik
                try:
                    print('Arrival Time {0:%Y/%m/%d %H:%M:%S} at Wind'.format(i))
                    print('Predicted Arrival Time at {2} {0:%Y/%m/%d %H:%M:%S}, Distance = {1:4.1f}km'.format(themis_pr,themis_d,esp.upper()))
                    print('Actual Arrival Time at {2} {0:%Y/%m/%d %H:%M:%S}, Offset (Pred.-Act.) = {1:4.2f}s'.format(itind,themis_dt-atval,esp.upper()))
                except:
                    continue

                #Use wind parameters to predict shock location 2018/04/25 J. Prchlik
                th_yval = t_mat.loc[i,:].SPEED
                th_xval = mdates.date2num(themis_pr)
                rl_xval = mdates.date2num(itind)
                    
                #plot parameters from wind prediction 2018/05/17 J. Prchlik
                ax_th.scatter(th_xval,th_yval,color='blue',label=None)
                #change to line at wind 2018/05/17
                #Add predicted THEMIS plot
                ax_th.annotate('Event {0:1d} at {1}'.format(j+1,esp.upper()),xy=(th_xval,th_yval),xytext=(th_xval,th_yval+50.),
                          arrowprops=dict(facecolor='purple',shrink=0.005))
                ###Add Actual to THEMIS plot 2018/05/03 J. Prchlik
                ##ax_th.annotate('Event {0:1d} at {1}'.format(j+1,esp.upper()),xy=(rl_xval,th_yval),xytext=(rl_xval,th_yval-50.),
                ##          arrowprops=dict(facecolor='red',shrink=0.005))
                #store speed and time values
                event_plot.append([th_xval,th_yval])
                event_plot.append([rl_xval,th_yval])

            #put values in new dataframe
            #for l in range(len(col_add)):
            #    frm_vs.loc[i,col_add[l]] = add_lis[l] 
        
        #turn big lis into numpy array
        #I don't need to do this 2018/03/15 J. Prchlik
        #big_lis = np.array(big_lis)
        
        
        fig.autofmt_xdate()
                        
        #Puff up y-limit 2018/05/03 J. Prchlik
        ylims = np.array(fax[2,0].get_ylim())
        #if ylim min less than 0 set to 250
        if ylims[0] < 0:
            ylims[0] = 250. 

        yrang = abs(ylims[1]-ylims[0])
        fax[2,0].set_ylim([ylims[0],ylims[1]+.1*yrang])
        #Save time warping plot
        fig.savefig('../plots/bou_{0:%Y%m%d_%H%M%S}.png'.format(pd.to_datetime(start_t)),bbox_pad=.1,bbox_inches='tight')
        
        #set up date and time ranges 2018/05/03 J. Prchlik
        event_plot = np.array(event_plot)
        min_val = event_plot.min(axis=0)
        max_val = event_plot.max(axis=0)
        rng_val = max_val-min_val
    
       
        #save resulting THEMIS plot 2018/04/25 J. Prchlik
        xlims = np.array(ax_th.get_xlim())
        ax_th.set_ylim([ylims[0],ylims[1]+.1*yrang])

        #include earth pad offset in time range 2018/05/04 J. Prchlik 
        xlims += self.pad_earth.total_seconds()/24./3600.

        #Add 10% padding around plot time window for events
        #increase time range if needed
        test_xmin = min_val[0]-.1*rng_val[0]
        test_xmax = max_val[0]+.1*rng_val[0]
        if test_xmin < xlims[0]:
            xlims[0] = test_xmin
        if test_xmax > xlims[1]:
            xlims[1] = test_xmax


        #use pretty axis and save
        ax_th.set_xlim(xlims)
        ax_th.legend(loc='best',frameon=False)
        fig_th.savefig('../plots/themis_pred_{0:%Y%m%d_%H%M%S}.png'.format(pd.to_datetime(start_t)),bbox_pad=.1,bbox_inches='tight')
        
        andir = '../plots/boutique_ana/'
        
        
        



        #Do not run animation sequence if asked to stop and return with just parameters
        #Returns vx,vy,vz,tvals,Vmag,Vnoraml,X,Y,Z
        #Switched to self.event_dict dictionary 2018/04/24 J. Prchlik
        if justparm: return 
        
        
        #sim_date =  pd.date_range(start=start_t,end=end_t,freq='60S')
        #switched to Wind index for looping and creating figures 2018/03/15
        sim_date = plsm['Wind'][start_t:end_t].index[::10]
        
        for i in sim_date:
            #list of colors
            cycol = cycle(['blue','green','red','cyan','magenta','black','teal','orange'])
        
            #Create figure showing space craft orientation
            ofig, oax = plt.subplots(nrows=2,ncols=2,gridspec_kw={'height_ratios':[2,1],'width_ratios':[2,1]},figsize=(18,18))
            tfig =  plt.figure(figsize=(18,18))
            tax = tfig.add_subplot(111, projection='3d')
            
            #set orientation lables
            oax[1,1].axis('off')
            oax[0,0].set_title('{0:%Y/%m/%d %H:%M:%S}'.format(i),fontsize=20)
            oax[0,0].set_xlabel('X(GSE) [R$_\oplus$]',fontsize=20)
            oax[0,0].set_ylabel('Z(GSE) [R$_\oplus$]',fontsize=20)
            oax[0,1].set_xlabel('Y(GSE) [R$_\oplus$]',fontsize=20)
            oax[0,1].set_ylabel('Z(GSE) [R$_\oplus$]',fontsize=20)
            oax[1,0].set_xlabel('X(GSE) [R$_\oplus$]',fontsize=20)
            oax[1,0].set_ylabel('Y(GSE) [R$_\oplus$]',fontsize=20)
        
            #set 3d axis label 2018/03/13
            tax.set_xlabel('X(GSE) [R$_\oplus$]',fontsize=20)
            tax.set_ylabel('Y(GSE) [R$_\oplus$]',fontsize=20)
            tax.set_zlabel('Z(GSE) [R$_\oplus$]',fontsize=20)
           
        
            ###########################################################
            ###########################################################
            #Added to get current plane orientation for Wind 2018/03/05
            tvals = [] 
            xvals = [] 
            yvals = [] 
            zvals = [] 
        
            #loop over all craft and populate time and position arrays
            for c in craft:
                #Get closest index value location
                ii = plsm[c].GSEx.dropna().index.get_loc(i,method='nearest')
        
                #convert index location back to time index
                it = plsm[c].GSEx.dropna().index[ii]
        
                #append craft values onto time and position arrays
                #changed to min values 2018/03/12 J. Prchlik
                itval = plsm[c+'_offset'].loc[i,'offsets']
                if isinstance(itval,pd._libs.tslib.Timedelta):
                    tvals.append(itval.total_seconds())
                elif isinstance(itval,pd.Series):
                    tvals.append(min(itval,key=abs).total_seconds())
                xvals.append(np.mean(plsm[c].loc[it,'GSEx']))
                yvals.append(np.mean(plsm[c].loc[it,'GSEy']))
                zvals.append(np.mean(plsm[c].loc[it,'GSEz']))
        
            #Covert arrays into numpy arrays and flip sign of offset
            tvals = np.array(tvals)
            xvals = np.array(xvals) 
            yvals = np.array(yvals) 
            zvals = np.array(zvals) 
        
            #use positions and vectors to get a solution for plane velocity
            pm  = np.matrix([xvals[1:]-xvals[0],yvals[1:]-yvals[0],zvals[1:]-zvals[0]]).T #coordinate of craft 1 in top row
            tm  = np.matrix(tvals[1:]).T # 1x3 matrix of time (wind-spacecraft)
            vna,vn,vm = solve_plane(pm,tm)
            #vna = np.linalg.solve(pm,tm) #solve for the velocity vectors normal
            #vn  = vna/np.linalg.norm(vna)
            #vm  = 1./np.linalg.norm(vna) #get velocity magnitude
            
            #store vx,vy,vz values
            vx,vy,vz = vm*np.array(vn).ravel()
        
            px = xvals[0]
            py = yvals[0]
            pz = zvals[0]
            #solve for the plane at time l
            #first get the points
            ps = np.matrix([[px],[py],[pz]])
        
        
            #switched to solve_coeff function 2018/04/24 J. Prchlik
            #solve the plane equation for d
            #d = float(vn.T.dot(ps))
            #print('###################################################')
            #print('NEW solution')
            #scale the coefficiecnts of the normal matrix for distance
            a,b,c,d = solve_coeff(ps,vn)
            #coeff = vn*pm
            #a = float(coeff[0])
            #b = float(coeff[1])
            #c = float(coeff[2])
            #d = float(coeff.T.dot(ps))
            #print(a,b,c,d)
            #print('###################################################')
            #get the wind plane values for given x, y, or z
            counter = np.linspace(-1e10,1e10,5)
            windloc = np.zeros(counter.size)
        
            #+/- range for plane
            rng = np.linspace(-1.e10,1.e10,100)
        
            #set off axis values to 0
            zvalsx = -(a*(px+rng)+b*(py+rng)-d)/c
            zvalsy = -(a*(px+rng)+b*(py+rng)-d)/c
            yvalsx = -(a*(px+rng)+c*(pz+rng)-d)/b
            #plot 2d plot
            oax[0,0].plot((px+rng)/Re,zvalsx/Re,color='gray',label="Current")
            oax[1,0].plot((px+rng)/Re,yvalsx/Re,color='gray',label=None)
            oax[0,1].plot((py+rng)/Re,zvalsy/Re,color='gray',label=None)
        
        
            ###########################################################
            ###########################################################
        
            #add fancy_plot to 2D plots
            for pax in oax.ravel(): fancy_plot(pax)
        
            #set labels for 3D plot
            tax.set_title('{0:%Y/%m/%d %H:%M:%S}'.format(i),fontsize=20)
        
        
            #Add radially propogating CME shock front    
            for p,l in enumerate(top_vs.index):
                #color to use
                cin = next(cycol)
                cur = 'event_{0:1d}'.format(p+1)
                vx = self.event_dict[cur]['vx']
                vy = self.event_dict[cur]['vy']
                vz = self.event_dict[cur]['vz']
                vm = self.event_dict[cur]['vm']
                vn = self.event_dict[cur]['vn']
                #Wind coordinates
                px = self.event_dict[cur]['wind_px']
                py = self.event_dict[cur]['wind_py']
                pz = self.event_dict[cur]['wind_pz']
                #timeself.event_dict difference between now and event
                dt = (i-l.to_pydatetime()).total_seconds()


                #theta normal angle
                theta = float(np.arctan(vn[2]/np.sqrt(vn[0]**2+vn[1]**2)))*180./np.pi
        
                #leave loop if shock is not within 2 hours therefore do not plot
                if np.abs(dt) > 2.*60.*60: continue
        
        
                #solve for the plane at time l
                #first get the points
                ps = np.matrix([[vx],[vy],[vz]])*dt+np.matrix([[px],[py],[pz]])
        
                #get the magentiude of the position
                pm  = float(np.linalg.norm(ps))
                
        
                #Switched to solve_coeff function 2018/04/24 J. Prchlik
                a,b,c,d = solve_coeff(ps,vn)
                #solve the plane equation for d
                #d = float(vn.T.dot(ps))
                ##print('###################################################')
                ##print('NEW solution')
                ##scale the coefficiecnts of the normal matrix for distance
                #coeff = vn*pm
                #a = float(coeff[0])
                #b = float(coeff[1])
                #c = float(coeff[2])
                #d = float(coeff.T.dot(ps))
                #print(a,b,c,d)
                #print('###################################################')
        
                #Switch to line 2018/03/15 J. 4rchlik
                ##set up arrays of values
                ##xvals = vx*dt+px
                ##yvals = vy*dt+py
                ##zvals = vz*dt+pz
        
                ##get sorted value array
                ##xsort = np.argsort(xvals)
                ##ysort = np.argsort(yvals)
                ##zsort = np.argsort(zvals)
        
                #get the plane values for given x, y, or z
                counter = np.linspace(-1e10,1e10,5)
                windloc = np.zeros(counter.size)
        
                #set off axis values to 0
                zvalsx = -(a*counter-d)/c
                zvalsy = -(b*counter-d)/c
                yvalsx = -(a*counter-d)/b
        
        
                #get shock Np value (use bfill to get values from the next good Np value) 
                np_df = plsm[trainer+'_offset'].Np.dropna()
                np_vl = np_df.index.get_loc(l,method='bfill')
                np_op = np_df.iloc[np_vl]
                
        
                #oax[0,0].text((vx*dt+px)[0],(vz*dt+pz)[0],plsm[trainer+'_offset'].loc[i,'Np'].dropna().min(),color='black')
                #oax[1,0].text((vx*dt+px)[0],(vy*dt+py)[0],plsm[trainer+'_offset'].loc[i,'Np'].dropna().min(),color='black')
                #oax[0,1].text((vy*dt+py)[0],(vz*dt+pz)[0],plsm[trainer+'_offset'].loc[i,'Np'].dropna().min(),color='black')
                #No need to solve for plane because I have values from the normal vector 2018/03/15 J. Prchlik
                #####plot 3d plot
                #####fit plane
                #####c, normal = fitPlaneLTSQ(xvals,yvals,zvals)
                #####try different way to fit plane
                ####A = np.c_[xvals,yvals,np.ones(xvals.size)]
                #####get cooefficience
                ####C,_,_,_ = scipy.linalg.lstsq(A,zvals)
                #create center point (use Wind)
                #point = np.array([xvals[0], yvals[0], zvals[0]])
                ##solve for d in a*x+b*y+c*z+d = 0
                #d = -point.dot(normal) #create normal surface
                #create mesh grid
                #Get max and min values
                maxx = 1900000.
                maxy = 300000.
                minx = 1200000.
                miny = -600000.
        
                #make x and y grids
                xg = np.array([minx,maxx])
                yg = np.array([miny,maxy])
        
                # compute needed points for plane plotting
                xt, yt = np.meshgrid(xg, yg)
                #zt = (-normal[0]*xt - normal[1]*yt - d)*1. / normal[2]
                #create z surface
                #switched to exact a,b,c from above 2018/03/15
                zt = -(a*xt+d*yt-d)/c
        
                #plot surface
                tax.plot_surface(xt/Re,yt/Re,zt/Re,color=cin,alpha=.5)
        
                #get sorted value array
                #xvals = xt.ravel()
                #yvals = yt.ravel()
                #zvals = zt.ravel()
                #zvals = xvals*C[0]+yvals*C[1]+C[2]
                xsort = np.argsort(xvals)
                ysort = np.argsort(yvals)
                zsort = np.argsort(zvals)
        
                #plot 2d plot
                oax[0,0].plot(counter/Re,zvalsx/Re,color=cin,
                              label='Shock {0:1d}, N$_p$ = {1:3.2f} cc, t$_W$={2:%H:%M:%S}, $|$V$|$={3:4.2f} km/s, {5} = {4:3.2f}'.format(p+1,np_op,l,vm,-theta,r'$\theta_\mathrm{Bn}$').replace('cc','cm$^{-3}$'))
                oax[1,0].plot(counter/Re,yvalsx/Re,color=cin,label=None)
                oax[0,1].plot(counter/Re,zvalsy/Re,color=cin,label=None)
        
        
            #get array of x,y,z spacecraft positions
            #spacraft positions
            for k in craft:
                #Get closest index value location
                ii = plsm[k].GSEx.dropna().index.get_loc(i,method='nearest')
                #convert index location back to time index
                it = plsm[k].GSEx.dropna().index[ii]
        
                oax[0,0].scatter(plsm[k].loc[it,'GSEx']/Re,plsm[k].loc[it,'GSEz']/Re,marker=marker[k],s=80,color=color[k],label=k)
                oax[1,0].scatter(plsm[k].loc[it,'GSEx']/Re,plsm[k].loc[it,'GSEy']/Re,marker=marker[k],s=80,color=color[k],label=None)
                oax[0,1].scatter(plsm[k].loc[it,'GSEy']/Re,plsm[k].loc[it,'GSEz']/Re,marker=marker[k],s=80,color=color[k],label=None)
                tax.scatter(plsm[k].loc[it,'GSEx']/Re,plsm[k].loc[it,'GSEy']/Re,plsm[k].loc[it,'GSEz']/Re,
                            marker=marker[k],s=80,color=color[k],label=None)
        
        
            #set static limits from orbit maximum from SOHO file in Re
            #z limits
            z_lim = np.array([-1.,1])*240.
            oax[0,0].set_ylim(z_lim)
            oax[0,1].set_ylim(z_lim)
            tax.set_zlim(z_lim)
            #xlimits
            x_lim = np.array([-1.,1])*240.0+225.
            oax[0,0].set_xlim(x_lim)
            oax[1,0].set_xlim(x_lim)
            tax.set_xlim(x_lim)
            #y limits
            y_lim = np.array([-1.,1])*120.0
            oax[0,1].set_xlim(y_lim)
            oax[1,0].set_ylim(y_lim)
            tax.set_ylim(y_lim)
            
            oax[0,0].legend(loc='upper right',frameon=False,scatterpoints=1)
        
        
            #rotate y,z y axis tick labels
            for tick in oax[0,1].get_xticklabels():
                tick.set_rotation(45)
        
            #Save spacecraft orientation plots
            ofig.savefig(andir+'event_orientation_{0:%Y%m%d_%H%M%S}.png'.format(i.to_pydatetime()),bbox_pad=.1,bbox_inches='tight')
            ofig.clf()
            #save 3d spacecraft positions
            tfig.savefig(andir+'d3/event_orientation_3d_{0:%Y%m%d_%H%M%S}.png'.format(i.to_pydatetime()),bbox_pad=.1,bbox_inches='tight')
            tfig.clf()
            plt.close()
            plt.close()
    

