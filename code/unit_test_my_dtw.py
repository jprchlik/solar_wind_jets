import pandas as pd
import matplotlib.pyplot as plt
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

    import mlpy
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
    my_dtw4 = mtr.dtw_plane(start_t4,end_t4,nproc=4,penalty=True,events=7,par=['Bt'],speed_pen=500,mag_pen=100.2)
    my_dtw4.init_read()
    my_dtw4.dtw()



    x1 = np.array(my_dtw4.plsm['Wind'].Bt.ffill().bfill().values,dtype=np.double)
    x2 = np.array(my_dtw4.plsm['DSCOVR'].Bt.ffill().bfill().values,dtype=np.double)


    return x1,x2,my_dtw4




if __name__ == '__main__':

    #test_1,test_2,cost_m = test_dtw()
    mlpy_example()
    x1,x2,my_dtw4 = test_dtw_example()

    p1,p2,cost = md.dtw_path_single(x2,x1,2700,334,0.0,1.1,1.0,1)
    #p1,p2,cost = md.dtw_path_single(x2,x2],2700,2700/2,0.0,0.01,1)

    fig, ax =plt.subplots()
    ax.imshow(cost,extent=[0,x2.size,0,x1.size],origin='lower')
    #ax.imshow(cost,extent=[0,x2.size,0,x2[::2].size],origin='lower')
    ax.plot(p1,p2,'--',color='black')
    plt.show()
