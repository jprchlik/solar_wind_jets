from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn import metrics

#from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict


import model_time_range as mtr
from fancy_plot import fancy_plot
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd


def get_x_difference(df,diff_v):


    new = ['diff_'+i for i in diff_v] 
    df[new] = df[diff_v].diff().bfill()
    diff_v = diff_v+new

    return df,diff_v

#create an equation to wieght the observation of the form y = b*(x-a)^2+c
def dtw_wei(x,t0,b=0.3,c=1):
    return b*(x-t0.value)**2+c

#Wind to get DTW solution
window = pd.to_timedelta(1.*3600.,unit='s')

#Setup format for datetime string to pass to my_dtw later
dfmt = '{0:%Y/%m/%d %H:%M:%S}'

#twind = pd.to_datetime('2016/12/20 15:38:38')    
#twind = pd.to_datetime('2016/12/21 08:43:12')    
#twind = pd.to_datetime('2016/07/28 13:25:21')    
#twind = pd.to_datetime('2017/07/01 16:55:03')    
#twind = pd.to_datetime('2016/10/12 21:16:14')    
#twind = pd.to_datetime('2016/12/09 04:45:29')
twind = pd.to_datetime('2016/12/21 08:43:12')    




################################################
#
#READ IN DATA
#
#
################################################

#################################################
#Training set
#################################################
start_t = dfmt.format(twind-3.5*window)
#For the 2016/12/21 event
end_t = dfmt.format(twind+3.5*window)
end_t = dfmt.format(twind+3.5*window)
#2016/12/09 04:45:29
#my_dtw = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B','THEMIS_C'],penalty=False,events=2)
#training set of data useing 1 time range
my_dtw_train = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B'],penalty=False,events=2)
my_dtw_train.init_read()


#################################################
#Test set
#################################################
start_t = dfmt.format(twind-4.5*window)
#For the 2016/12/21 event
end_t = dfmt.format(twind+4.5*window)
#test data set using another time range
my_dtw_tests = mtr.dtw_plane(start_t,end_t,nproc=4,earth_craft=['THEMIS_B'],penalty=False,events=2)
my_dtw_tests.init_read()



#################################################
##
#Configure DATA#
##
##
#################################################

#################################################
#Training set
#################################################
#X values to use for training
x_vals_i = ['SPEED','Bx','By','Bz']#,'Vth','Np']

#fill all Nans in data series with forward values first and then back fill the first times
my_dtw_train.plsm['Wind'] = my_dtw_train.plsm['Wind'].ffill().bfill()

#Get and record differences and add to x_vals
my_dtw_train.plsm['Wind'],x_vals = get_x_difference(my_dtw_train.plsm['Wind'],x_vals_i) 


#set up variables for training set
X_train = my_dtw_train.plsm['Wind'][x_vals].values

#Include a time normalization
normer_train = my_dtw_train.plsm['Wind']['Time_pls'].values.astype('float64')

min_train = normer_train.min()
max_train = normer_train.max()

#create normalization values
normer_train = dtw_wei(normer_train,twind,b=5./((max_train-min_train)/2.)**2.,c=1.)

#normer_train += -min_train
#normer_train /= max_train
#normer_train += 1.
normer_train = np.outer(normer_train,np.ones(X_train.shape[1]))

#include a time normalization factory
X_train *= normer_train

y_train = my_dtw_train.plsm['Wind'].index.values

#################################################
#Test set
#################################################

for j in ['DSCOVR','ACE','SOHO']:
    #fill all Nans in data series with forward values first and then back fill the first times
    my_dtw_tests.plsm[j] = my_dtw_tests.plsm[j].ffill().bfill()
    #Get and record differences and add to x_vals
    my_dtw_tests.plsm[j],x_vals = get_x_difference(my_dtw_tests.plsm[j],x_vals_i) 
    
    
    
    #set up variables for test set
    X_tests = my_dtw_tests.plsm[j][x_vals].values#my_dtw_tests.plsm[j]['time_check'].values/my_dtw_tests.plsm[j]['time_check'].max()
    
    #Include a time normalization
    normer_tests = my_dtw_tests.plsm[j]['Time_pls'].values.astype('float64')
    
    
    #create normalization values
    normer_tests = dtw_wei(normer_tests,twind,b=5./((max_train-min_train)/2.)**2.,c=1.)
    normer_tests = np.outer(normer_tests,np.ones(X_tests.shape[1]))
    
    #include a time normalization factory
    X_tests *= normer_tests
    
    y_tests = my_dtw_tests.plsm[j].index.values
    
    
    
    #get multi-parameter dtw solution
    path, sim = metrics.dtw_path(X_train, X_tests)
    
    #convert path into a numpy array
    m = np.array(zip(*path))
    fig, nax = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=(12,12))
    
    #x_vals = ['SPEED']+x_vals
    for i,ax in enumerate(nax.flatten()):
        parm = x_vals[i]
        ax.plot(y_train[m[0,:]],my_dtw_train.plsm['Wind'][parm].values[m[0,:]])
        ax.plot(y_train[m[0,:]],my_dtw_tests.plsm[j][parm].values[m[1,:]])
        
        ax.set_ylabel(parm)
        ax.set_xlabel('Time [UTC]')
        fancy_plot(ax)
#ax.plot(my_dtw_train.plsm['Wind'].index.values.astype('float64'),X_train[:,0])
#ax.plot(my_dtw_tests.plsm[j].index.values.astype('float64'),X_tests[:,0])
#ax.set_yscale('log')

plt.show()


#Missing KERAS 2018/07/25 J. Prchlik
#Try matching shaplets
####shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],
####                                                       ts_sz=X_train.shape[1],
####                                                       n_classes=len(set(y_train)),
####                                                       l=0.1,
####                                                       r=2)
####
####
####shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
####                        optimizer=Adagrad(lr=.1),
####                        weight_regularizer=.01,
####                        max_iter=50,
####                        verbose_level=0)
####
####
####shp_clf.fit(X_train, y_train)
####predicted_locations = shp_clf.locate(X_test)
####
####
####
####test_ts_id = 0
####plt.figure()
####plt.title("Example locations of shapelet matches (%d shapelets extracted)" % sum(shapelet_sizes.values()))
####plt.plot(X_test[test_ts_id].ravel())
####for idx_shp, shp in enumerate(shp_clf.shapelets_):
####    t0 = predicted_locations[test_ts_id, idx_shp]
####    plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)
####
####plt.tight_layout()
####plt.show()
####
#Nearest Neighbors not working in any real way yet 2018/07/25 J. Prchlik
#do the DTW fit test
####knn = KNeighborsTimeSeries(n_neighbors=15,metric='dtw')
####knn.fit(X_train,y_train)
####dists, ind = knn.kneighbors(X_tests)
####print("1. Nearest neighbour search")
####print("Computed nearest neighbor indices (wrt DTW)\n", ind)
####print("First nearest neighbor class:", y_tests[ind[:, 0]])
####
####
##### Nearest neighbor classification
####knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
####knn_clf.fit(X_train, y_train)
####predicted_labels = knn_clf.predict(X_tests)
####print("\n2. Nearest neighbor classification using DTW")
####print("Correct classification rate:", accuracy_score(y_tests, predicted_labels))

