multi\_dtw module
=================

.. automodule:: multi_dtw
    :members:
    :undoc-members:
    :show-inheritance:


.. code-block:: python

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


.. code-block:: python

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
