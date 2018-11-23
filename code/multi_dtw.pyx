import numpy as np 

DTYPE = np.double
uTYPE = np.long


def dtw_single(double [:] var_1, double [:] var_2,long lim_1, long lim_2, double pen_r):
    """
    Single value DTW cost matrix

    Parameters
    ----------
    var_1: np.array
        Array of values to DTW
    var_2: np.array
        Array of values to DTW
    lim_1: int
        Width in pixels of the DTW solution check before applying penalty for variable 1.
    lim_2: int
        Width in pixels of the DTW solution check before applying penalty for variable 2.
    pen_r: float
        Penalty value to apply when values exceed lim_r



    Returns
    ---------
    diff_arr: np.array
        A 2D array cost matrix with penalty included
    """

    #Define variables used for looping
    cdef long var_1_x = var_1.size   
    cdef long var_2_x = var_2.size   
    cdef long x1,x2
    cdef double [3] back_cells
    cdef double [:,:] diff_arr = np.zeros((var_1_x,var_2_x),dtype=DTYPE)
    cdef double m,dx_1,dx_2,penalty,cumulative,w_val = 0.


    #get the offset slope to allow for penalties when there is too much DTW compression
    #assumes they have the same start time
    m = float(var_1_x)/float(var_2_x)   

    #loop over time variable for input number 1
    for x1 in range(var_1_x):
        #loop over time varaible for input number 2
        for x2 in range(var_2_x):
           
            #add penalty
            penalty = 0.
            #check to see if time index is outside compression radius
            dx_1 = abs(m*x2-x1)
            dx_2 = abs((1./m)*x1-x2)

            #if either radius is larger than a limiting radius radius add a penalty
            if ((dx_1 >= lim_1) | (dx_2 >= lim_2)):
                penalty = pen_r

            #You really don't want this 2018/11/15 J. Prchlik
            #near the start and end remove the penalty
            #if ((x1 <= lim_r) | (x2 <= lim_r) | (var_1_x-x1 <=lim_r) | (var_2_x-x2 <= lim_r)):  
            #    penalty = 0.

            #cumlative distance values
            if x1 > 0:
                back_cells[0] = diff_arr[x1-1,x2]
            if x2 > 0:
                back_cells[1] = diff_arr[x1,x2-1]
            if ((x1 > 0) & (x2 > 0)):
                back_cells[2] = diff_arr[x1-1,x2-1]
 
            #Start at 0 value
            if ((x1 < 1) & (x2 < 1)):
                cumulative = 0
            elif ((x1 < 1) & ( x2 > 0)):
                cumulative = diff_arr[x1,x2-1]
            elif ((x1 > 0) & (x2 < 1)):
                cumulative = diff_arr[x1-1,x2]
            else:
                #minimum cost from the previous cells
                cumulative = min(back_cells)
            
            #store the difference
            diff_arr[x1,x2] = abs(var_1[x1]-var_2[x2])+cumulative+penalty#*d_x/lim_r

    return diff_arr



def dtw_path_single(double [:] var_1, double [:] var_2,long lim_1, long lim_2, double pen_r, double comp_p,double rad_tol, long cost_matrix):
    """
    Single value DTW path solution. The code applies two different penalties. One for venturing outside at pixel window 
    set by lim_r, after which the penalty is pen_r*d_x/lim_r where d_x is the pixel difference between a line that bisects 
    the cost array (i.e. abs(m*x2-x1), where m is the the length of var_1 over the length of var_2. The next penalty is a 
    compression penalty, which we add at the number of compression elements greater than the standard cadence difference
    multiplied by ten percent of the uncompressed matrix cost of that element.

    Parameters
    ----------
    var_1: np.array
        Array of values to DTW
    var_2: np.array
        Array of values to DTW
    lim_1: int
        Width in pixels of the DTW solution check before applying penalty for variable 1.
    lim_2: int
        Width in pixels of the DTW solution check before applying penalty for variable 2.
    pen_r: float
        Penalty value to apply when values exceed lim_r
    comp_p: float
        Percent of cost_matrix[i,j] to scale when compression has occurred beyond time cadence differences.
    rad_tol: float
        The radius tolerance for when checking whether the data is too compressed as a fraction.
    cost_matrix: int
        Whether or not to return the cost matrix. Set cost_matrix = 1 to return the cost matrix, 
        any other integer value will not.

    Returns
    ---------
    diff_arr: np.array
        A 2D array 
    """
    cdef long var_1_x = var_1.size   
    cdef long var_2_x = var_2.size   
    cdef long max_len = max([var_1_x,var_2_x])
    cdef long min_len = min([var_1_x,var_2_x])
    cdef long i=var_1_x-1,j=var_2_x-1,l=1,start_ind,pad_s 
    cdef double rat_len = float(max_len)/float(min_len)
    cdef double[:,:] diff_arr = np.zeros((var_1_x,var_2_x),dtype=DTYPE)
    cdef long [:] path_var_1 = np.zeros(max_len*int(rat_len+1.),dtype=uTYPE)-999
    cdef long [:] path_var_2 = np.zeros(max_len*int(rat_len+1.),dtype=uTYPE)-999
    cdef double [:] path_val_1 = np.zeros(max_len*int(rat_len+1.),dtype=DTYPE)
    cdef double [:] path_val_2 = np.zeros(max_len*int(rat_len+1.),dtype=DTYPE)
    cdef double [:] var_1_d = np.zeros(var_1_x,dtype=DTYPE)
    cdef double [:] var_2_d = np.zeros(var_2_x,dtype=DTYPE)
    cdef double i_cost,j_cost,b_cost,comp_r,comp_s
    cdef double i_comp=0.,j_comp=0.


    #compression ratio 
    comp_r = float(var_2_x)/float(var_1_x)
    #current compression slope
    comp_s = comp_r
    #print('HERE')
    ##Make sure variables are double type
    #var_1_d = np.array(var_1,dtype=DTYPE)
    #var_2_d = np.array(var_2,dtype=DTYPE)
  
    #Pad to start looking for slope
    pad_s = int(round(float(min_len)*0.01))

    #get the difference array
    diff_arr = dtw_single(var_1,var_2,lim_1,lim_2,pen_r)

    #First variable input
    path_var_1[0] = i
    path_var_2[0] = j

    #calculate the path
    while (((i > 0) | (j > 0)) & (l < max_len*round(rat_len+1.))) :
        #i index already at zero then just decrease j index
        if i == 0:
            j = j -1 
        #j index already at zero then just decrease i index
        elif j == 0:
            i = i -1 
        else:
            #calculate the cost for changing each element 
            i_cost = diff_arr[i-1,j]
            j_cost = diff_arr[i,j-1]
            b_cost = diff_arr[i-1,j-1]

            #Calculate current slope of the DTW path
            if (((var_1_x-i) > lim_1+pad_s) | ((var_2_x-j) > lim_2+pad_s)):
                comp_s = 0.
                for r in range(l-pad_s,l):
                    comp_s += path_val_2[r]/path_val_1[r]
                comp_s /= pad_s

            #Add in compression penalty if i is compressed
            if (comp_s-comp_r)/comp_r < -rad_tol:
                j_cost = (comp_p*i_comp+1.)*j_cost
                #diff_arr[i,j-1] = j_cost
                #print('Compressed I',i)
            #Add in compression penalty if j is compressed
            if (comp_s-comp_r)/comp_r > rad_tol:
                i_cost = (comp_p*j_comp+1.)*i_cost
                #diff_arr[i-1,j] = i_cost
                #print('Compressed J',j)

            
            #caclulate the min
            m_cost = np.argmin([b_cost,i_cost,j_cost])

            #update i,j index in path
            #if ((i_cost == m_cost) & (j_cost ==m_cost)):
            #    j = j-1
            #    i = i-1
            #    #decompress i and j
            #    i_comp = 0.
            #    j_comp = 0.
            if m_cost == 1:
                i = i-1
                #Start counting the compression of j
                j_comp = j_comp+1.
                #decompress i
                i_comp = 0.
            elif m_cost == 2:
                j = j-1
                #Start counting the compression of i
                i_comp = i_comp+1.
                #decompress j
                j_comp = 0.
            else:
                j = j-1
                i = i-1
                #decompress i and j
                i_comp = 0.
                j_comp = 0.

        #Do not start appying compression penalty until you reach
        #a number of pixels beyond lim_r
        if (((var_1_x-i) < lim_1) & ((var_2_x-j) < lim_2)):
            i_comp = 0.
            j_comp = 0.

        #store the path variable
        path_var_1[l] = i
        path_var_2[l] = j
        #store the path value for calculating slope
        path_val_1[l] = float(i)
        path_val_2[l] = float(j)

        #update the path variable index 
        l = l+1
    #Add 0 0 to the last index
    path_var_1[l] = 0
    path_var_2[l] = 0

    #Remove all values after l
    path_var_1 = np.array(path_var_1)[:l]
    path_var_2 = np.array(path_var_2)[:l]
    #flip the order of the output array
    path_var_1 = path_var_1[::-1]
    path_var_2 = path_var_2[::-1]

    #Remove all values before the index location (They will all by -9999)
    start_ind = 0#int(max_len*round(rat_len+1.)-l)
             
    #remove all -999 values in return
    if cost_matrix == 1:
        return np.array(path_var_1[start_ind:]),np.array(path_var_2[start_ind:]),np.array(diff_arr)
    else:                          
        return np.array(path_var_1[start_ind:]),np.array(path_var_2[start_ind:])
