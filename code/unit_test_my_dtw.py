
import numpy as np
import multi_dtw as md
import time

def test_dtw():

    inp_1 = np.linspace(0,40,40000)
    inp_1 = np.vstack([inp_1,inp_1])
    inp_2 = np.linspace(0,40,20000)
    inp_2 = np.vstack([inp_2,inp_2])


    time_s = time.time()
    test_1, test_2,cost_m = md.dtw_path(inp_1,inp_2,100,0,1)
    time_e = time.time()
    print('Time to run DTW {0:2.1f}'.format(time_e-time_s))


    #test the program returns the answers in the correct order
    ind_1 = np.sum(np.abs(test_1-np.array(sorted(range(40000))[:-1])))
    ind_2 = np.sum(np.abs(test_2-np.array(sorted(range(20000)*2)[:-1])))
  
    #use the sum of 0 integers to confirm it is true
    assert ind_1 == 0
    assert ind_2 == 0


    #Return values for testing purposes
    return test_1,test_2,cost_m




if __name__ == '__main__':

    test_1,test_2,cost_m = test_dtw()