import pandas as pd
from fancy_plot import fancy_plot
import mlpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import model_time_range as mtr
import numpy as np
import multi_dtw as md
import time

def test_dtw():
    """
    Tests the accuracy of my 1D DTW solution

    """

    #samples for input1 and input2
    sam_1 = 4000
    sam_2 = 2000

    inp_1 = np.linspace(0,40,sam_1)
    #inp_1 = np.vstack([inp_1,inp_1])
    inp_2 = np.linspace(0,40,sam_2)
    #inp_2 = np.vstack([inp_2,inp_2])


    time_s = time.time()
    test_1, test_2,cost_m = md.dtw_path_single(inp_1,inp_2,2,2,100,1.0,1.0,1)
    time_e = time.time()
    print('Time to run DTW {0:2.1f}s'.format(time_e-time_s))


    #test the program returns the answers in the correct order
    ind_1 = np.sum(np.abs(test_1-np.array(sorted(range(sam_1)))))
    ind_2 = np.sum(np.abs(test_2-np.array(sorted(range(sam_2)*2))))
  
    #use the sum of 0 integers to confirm it is true
    assert ind_1 == 0
    assert ind_2 == 0


    #Return values for testing purposes
    return test_1,test_2,cost_m


def mlpy_example():

    x = np.array([0,0,0,0,1,1,2,2,3,2,1,1,0,0,0,0],dtype=np.double)
    y = np.array([0,0,1,1,2,2,3,3,3,3,2,2,1,1,0,0],dtype=np.double)
    test_1, test_2,cost_m = md.dtw_path_single(x,y,200,200,100,5.0,5.0,1)

    dist, cost, path = mlpy.dtw_std(x, y, dist_only=False)
    true_1 = path[0]
    true_2 = path[1]
    #test the program returns the answers in the correct order
    ind_1 = np.sum(np.abs(test_1-true_1))
    ind_2 = np.sum(np.abs(test_2-true_2))

    #print(cost_m-cost)
    #fig, ax =plt.subplots(ncols=2,sharex=True,sharey=True)
    #ax[0].imshow(cost_m.T,extent=[0,x.size,0,y.size],origin='lower')
    #ax[1].imshow(cost.T  ,extent=[0,x.size,0,y.size],origin='lower')
    ##ax.imshow(cost,extent=[0,x2.size,0,x2[::2].size],origin='lower')
    #ax[0].plot(test_1,test_2,'--',color='black')
    #ax[1].plot(true_1,true_2,'.-',color='red')
    #plt.show()

  
    #use the sum of 0 integers to confirm it is true
    assert ind_1 == 0
    assert ind_2 == 0
    assert np.allclose(cost_m,cost)




def test_dtw_example():
    """
    Test using code
    """
    #Wind to get DTW solution
    window = pd.to_timedelta(1.*3600.,unit='s')
    
    
    
    #Setup format for datetime string to pass to my_dtw later
    dfmt = '{0:%Y/%m/%d %H:%M:%S}'


    #twind4 = pd.to_datetime('2016/12/09 04:45:29')
    #start_t4 = dfmt.format(twind4-2.5*window)
    #end_t4 = dfmt.format(twind4+3.5*window)
    twind4 = pd.to_datetime('2016/12/21 08:43:12')
    start_t4 = dfmt.format(twind4-2.5*window)
    end_t4 = dfmt.format(twind4+3.5*window)
    #my_dtw4 = mtr.dtw_plane(start_t4,end_t4,nproc=4,penalty=True,events=7,earth_craft=['THEMIS_B'],par=['Bt'],speed_pen=500,mag_pen=100.2)
    my_dtw4 = mtr.dtw_plane(start_t4,end_t4,nproc=4,penalty=True,events=7,par=['Bt'],earth_craft=['THEMIS_B'],speed_pen=500,mag_pen=100.2)
    my_dtw4.init_read()
    my_dtw4.iterate_dtw()

    #my_dtw4.pred_earth()
    #mtr.omni_plot(my_dtw4)

    sc1 = 'Wind'
    sc2 = 'SOHO'

    x1 = np.array(my_dtw4.plsm[sc1].SPEED.ffill().bfill().values,dtype=np.double)
    x2 = np.array(my_dtw4.plsm[sc2].SPEED.ffill().bfill().values,dtype=np.double)


    #Example DTW plot
    p1,p2,cost1 = md.dtw_path_single(x1,x2,300,30,500.0,0.0,0.5,1)
    #p1,p2,cost = md.dtw_path_single(x2,x2],2700,2700/2,0.0,0.01,1)
    #mlpy example path
    dist, costa, path = mlpy.dtw_std(x1,x2, dist_only=False)
    pa, pb = path[0],path[1]


    #create multi panel diagnostic plot 2018/11/26 J. Prchlik
    fig, ax = plt.subplots(nrows=2,ncols=2,gridspec_kw={'height_ratios':[2,1],'width_ratios':[1,2]},figsize=(8,8))
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    #turn off bottom left axis
    ax[1,0].axis('off')
    lims = mdates.date2num([my_dtw4.plsm[sc1].index.min(),my_dtw4.plsm[sc1].index.max(),my_dtw4.plsm[sc2].index.min(),my_dtw4.plsm[sc2].index.max()])
    v_max,v_min = np.percentile(costa,[95,15])
    ax[0,1].imshow(costa,extent=lims,origin='lower',cmap=plt.cm.gray.reversed(),vmin=v_min,vmax=v_max,aspect='auto')
    #ax.imshow(cost,extent=[0,x2.size,0,x2[::2].size],origin='lower')
    ax[0,1].plot(my_dtw4.plsm[sc1].iloc[p1,:].index,my_dtw4.plsm[sc2].iloc[p2,:].index,'--',color='black')
    ax[0,1].plot(my_dtw4.plsm[sc1].iloc[pa,:].index,my_dtw4.plsm[sc2].iloc[pb,:].index,'-',color='red')
    ax[0,1].xaxis_date()
    ax[0,1].yaxis_date()
    date_format = mdates.DateFormatter('%H:%M')


    #plot the plasma values on the off axes
    ax[1,1].plot(my_dtw4.plsm[sc1].index,x1,color='blue')
    ax[0,0].plot(x2,my_dtw4.plsm[sc2].index,color='teal')
 
    #set up axis formats 
    ax[1,1].xaxis_date()
    ax[0,0].yaxis_date()

    #force limits to be the same as the cost matrxi
    ax[1,1].set_xlim(lims[:2])
    ax[0,0].set_ylim(lims[2:])
    
    #Format the printed dates
    ax[1,1].xaxis.set_major_formatter(date_format)
    ax[0,0].yaxis.set_major_formatter(date_format)
    
    #Add label time 
    ax[1,1].set_xlabel(sc1+' Time [UTC]')
    ax[0,0].set_ylabel(sc2+' Time [UTC]')

    #Add label for Speeds
    ax[1,1].set_ylabel('Flow Speed [km/s]')
    ax[0,0].set_xlabel('Flow Speed [km/s]')

    #turn off y-tick labels in center plot
    ax[0,1].set_xticklabels([])
    ax[0,1].set_yticklabels([])

    #set Wind and SOHO to have the same plasma paramter limits
    pls_lim = [420.,675.]
    ax[0,0].set_xlim(pls_lim)
    ax[1,1].set_ylim(pls_lim)

    #copy y-axis labels from Wind plot to SOHO plot
    ax[0,0].set_xticks(ax[1,1].get_yticks())
    ax[0,0].set_xlim(pls_lim)
    ax[1,1].set_ylim(pls_lim)
    ##ax[0,0].set_xlabel('Flow Speed [km/s]')

    
    #clean up the axes with plasma data
    fancy_plot(ax[0,0])
    fancy_plot(ax[1,1])

    # This simply sets the x-axis data to diagonal so it fits better.
    #fig.autofmt_xdate()
    fig.savefig('../plots/example_dtw_path.png',bbox_pad=.1,bbox_inches='tight')
    fig.savefig('../plots/example_dtw_path.eps',bbox_pad=.1,bbox_inches='tight')
    return x1,x2,my_dtw4




if __name__ == '__main__':

    test_1,test_2,cost_m = test_dtw()
    mlpy_example()
    x1,x2,my_dtw4 = test_dtw_example()

