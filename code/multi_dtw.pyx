import numpy as np 

DTYPE = np.double
uTYPE = np.int

def dtw_grid(double [:,:] var_1, double [:,:] var_2,int lim_r, double pen_r):

    #Define variables used for looping
    cdef int var_1_x = var_1.shape[0]   
    cdef int var_1_y = var_1.shape[1]   
    cdef int var_2_x = var_2.shape[0]   
    cdef int var_2_y = var_2.shape[1]   
    cdef int x1,y1,y2
    cdef double [:,:,:] diff_arr = np.zeros((var_1_x,var_1_y,var_2_y),dtype=DTYPE)
    cdef float m,dy,penalty


    #check to make sure x and y have the same number of variables (not the samp time series length)
    if var_1_x != var_2_x:
        raise ValueError("var_1 and var_2 must have the same number of variables. Currently, var_1=%d, var_2=%d" % [var_1_x,var_2_x]) 


    #get the offset slope to allow for penalties when there is too much DTW compression
    #assumes they have the same start time
    m = var_1_y/var_2_y    

    #loop over x variables
    for x1 in range(var_1_x):
        #loop over time variable for input number 1
        for y1 in range(var_1_y):
            #loop over time varaible for input number 2
            for y2 in range(var_2_y):
               
                #add penalty
                penalty = 0.
                #check to see if time index is outside compression radius
                d_y = abs(m*y1-y2)

                #if d_y larger than the compression radius add a penalty
                if d_y >= lim_r:
                    penalty = pen_r

                #near the start and end remove the penalty
                if ((y1 <= lim_r) | (y2 <= lim_r) | (var_1_y-y1 <=lim_r) | (var_2_y-y2 <= lim_r)):  
                    penalty = 0.
                
                #store the difference
                diff_arr[x1,y1,y2] = abs(var_1[x1,y1]-var_2[x1,y2])+penalty

    return diff_arr

def dtw_path(double [:,:] var_1, double [:,:] var_2,int lim_r, double pen_r, int cost_matrix):

    cdef int var_1_x = var_1.shape[0]   
    cdef int var_1_y = var_1.shape[1]   
    cdef int var_2_x = var_2.shape[0]   
    cdef int var_2_y = var_2.shape[1]   
    cdef int i=var_1_y-1,j=var_2_y-1,l=0
    cdef int max_len = max([var_1_y,var_2_y])
    cdef int min_len = max([var_1_y,var_2_y])
    cdef float rat_len = float(max_len)/float(min_len)
    cdef double[:,:,:] diff_arr = np.zeros((var_1_x,var_1_y,var_2_y),dtype=DTYPE)
    cdef long [:] path_var_1 = np.zeros(max_len*int(rat_len+1.),dtype=uTYPE)-999
    cdef long [:] path_var_2 = np.zeros(max_len*int(rat_len+1.),dtype=uTYPE)-999
    cdef float i_cost,j_cost,b_cost

    #check to make sure x and y have the same number of variables (not the samp time series length)
    if var_1_x != var_2_x:
        raise ValueError("var_1 and var_2 must have the same number of variables. Currently, var_1=%d, var_2=%d" % [var_1_x,var_2_x]) 

    #get the difference array
    diff_arr = dtw_grid(var_1,var_2,lim_r,pen_r)

    #calculate the path
    while ((i > 0) & (j > 0) & (l < max_len*round(rat_len+1.))) :
        #i index already at zero then just decrease j index
        if i == 0:
            j = j -1 
        #j index already at zero then just decrease i index
        elif j == 0:
            i = i -1 
        else:
            #calculate the cost for changing each element 
            i_cost = sum(diff_arr[:,i-1,j])
            j_cost = sum(diff_arr[:,i,j-1])
            b_cost = sum(diff_arr[:,i-1,j-1])
            #caclulate the min
            m_cost = min([i_cost,j_cost,b_cost])

            #update i,j index in path
            if i_cost == m_cost:
                i = i-1
            elif j_cost == m_cost:
                j = j-1
            else:
                j = j-1
                i = i-1

        #store the path variable
        path_var_1[l] = i
        path_var_2[l] = j

        #update the path variable index 
        l = l+1

    
    if cost_matrix == 1:
        return np.array(path_var_1),np.array(path_var_2),np.array(diff_arr)
    else:
        return np.array(path_var_1),np.array(path_var_2)