
import numpy as np
import multi_dtw as md

def test_dtw():

    inp_1 = np.linspace(0,40,200)
    inp_1 = np.vstack([inp_1,inp_1])
    inp_2 = np.linspace(0,40,100)
    inp_2 = np.vstack([inp_2,inp_2])


    test_1, test_2,cost_m = md.dtw_path(inp_1,inp_2,100,.1,1)


    print(test_1)
    print(test_2)
  
    return test_1,test_2,cost_m




if __name__ == '__main__':

    test_1,test_2,cost_m = test_dtw()