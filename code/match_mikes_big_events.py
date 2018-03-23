import model_time_range as mtr
from scipy.io import readsav
from fancy_plot import fancy_plot
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd



mike_event = glob('../mikes_big_events/*idl')


#remove event with wrong julian date for now
mike_event.remove('../mikes_big_events/event_20170527.idl')
mike_event.remove('../mikes_big_events/event_20151024.idl')

#mike keys in the order I use them (Wind,DSCOVR,ACE,SOHO)
skey = ['WI','DS','AC','SO']

#positions to get from orbital array
pkey = ['TIME','X','Y','Z']

#Velocities from solution to get from mike array
vkey = ['TIME','V','VX','VY','VZ','N','W']


#Wind to get DTW solution
window = pd.to_timedelta(3600.,unit='s')


#create class which contains information on all spacecraft
my_dtw = mtr.dtw_plane('2016/07/10 00:00:00','2017/09/15 00:00:00',nproc=4)
my_dtw.init_read()  

#Setup format for datetime string to pass to my_dtw later
dfmt = '{0:%Y/%m/%d %H:%M:%S}'

#loop over all events and compare with my plane solution
for event in mike_event:


    #read in mike's save file
    mdat = readsav(event)

    #Wind shock time in JD and convert to datetime object
    twind= pd.to_datetime(np.sum(mdat['state']['TRANGE'][0])/2.,origin='julian',unit='D')


    
    #stored values
    boff = [] #time offsets
    xval = [] #x values
    yval = [] #y values
    zval = [] #z values
    xvel = [] #x velocity
    yvel = [] #y velocity
    zvel = [] #z velocity
    mvel = [] #magnitude of velocity
    nomv = [] #nrmal vector

    #loop over all spacecraft
    for j in skey:
        #offset time for all spacecraft and convert to seconds 
        toff = mdat['state']['tshifts'][0][j][0]*3600.*24.
        boff.append(toff)

        #time offset for given spacecraft 
        soff = twind+pd.to_timedelta(toff,unit='s')

        #position of spacecraft in temporary data array
        tdat = mdat['state']['data'][0][j][0]['O'][0][pkey][0]

        #create orbital pandas dataframe with tdat
        obit = pd.DataFrame(np.array([tdat[p] for p,q in enumerate(tdat)]).T,columns=pkey)
        obit['time_dt'] = pd.to_datetime(obit.TIME,unit='D',origin='julian')
        obit.set_index(obit.time_dt,inplace=True)

        #get nearest index for pandas dataframe position
        #Get closest index value location
        ii = obit.index.get_loc(soff,method='nearest')
        #convert index location back to time index
        it = obit.index[ii]
        #add spacecraft positions to lists
        xval.append(obit.loc[it,'X'])
        yval.append(obit.loc[it,'Y'])
        zval.append(obit.loc[it,'Z'])

        #position of spacecraft in temporary data array
        tdat = mdat['state']['data'][0][j][0]['P'][0][vkey][0]

        #create orbital pandas dataframe with tdat
        vdat = pd.DataFrame(np.array([tdat[p] for p,q in enumerate(tdat)]).T,columns=vkey)
        vdat['time_dt'] = pd.to_datetime(vdat.TIME,unit='D',origin='julian')
        vdat.set_index(vdat.time_dt,inplace=True)

        #get nearest index for pandas dataframe position
        #Get closest index value location
        ii = vdat.index.get_loc(soff,method='nearest')
        #convert index location back to time index
        it = vdat.index[ii]
        #add spacecraft positions to lists
        xvel.append(vdat.loc[it,'VX'])
        yvel.append(vdat.loc[it,'VY'])
        zvel.append(vdat.loc[it,'VZ'])
        mvel.append(vdat.loc[it,'V'])
        nomv.append(vdat.loc[it,'N'])



    #convert lists to numpy arrays
    boff = np.array(boff)
    xval = np.array(xval)
    yval = np.array(yval)
    zval = np.array(zval)
    xvel = np.array(xvel)
    yvel = np.array(yvel)
    zvel = np.array(zvel)
    mvel = np.array(mvel)
    nomv = np.array(nomv)


    #get my solution
    my_dtw.start_t = dfmt.format(twind-window)
    my_dtw.end_t = dfmt.format(twind+window)
    my_dtw.par = None
    big_arr = my_dtw.main()
    #big_arr = mtr.main(str(twind-window),str(twind+window))

    vn = np.array(big_arr[0][5]).T #The normal vectory for my solution
    tvals = big_arr[0][3]

    print('#######################################################')
    print('Wind EVENT AT',twind)
    print('Differnces in normals')
    print(vn,nomv)
    print('Difference in Time offsets')
    print(tvals,boff)
    print('#######################################################')
    
    
    #get time offset differences