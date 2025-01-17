#start unit testing
import unittest
import model_time_range as mtr
import numpy as np

#Test to make sure the plane solution works 2018/04/24 J. Prchlik
class check_solution(unittest.TestCase):
    """
    A class that checks that you will recover a propagating input plane wave

    """
    
    def test_plane_creation(self):
        #equation for plane 
        a,b,c,d = -1000.0,400.0,300.0,700000.0 #km
        #set wind position 
        x = 200.
        y = 0.
        z = -(a*x+b*y+d)/c
        wind_pos =  np.matrix([x,y,z])
        self.assertEqual(x*a+y*b+z*c+d,0)


    def test_velocity_mag(self):
        a,b,c,d = -1000.0,400.0,300.0,100000.0 #km
        #set wind position 
        x = 200.
        y = 0.
        z = -(a*x+b*y+d)/c
        wind_pos = np.matrix([x,y,z])

        #set really far away position
        xf = 1.2E8
        yf = -1.E3
        zf = -(a*xf+b*yf+d)/c

        #magnitude of velocity
        vm = 600. #km/s
        #get normal vector  
        p = np.matrix([a,b,c])
        vn = p/np.linalg.norm(p)

        #create a matrix of space craft posittons
        othr_pos = np.matrix([[500.,-1200.,700.],  #x values
                             [200.,400.,-1700.],#y values
                             [1500,-600,40]])    #z values


        #get magnitude of distance from plane measured at wind to other spacecraft
        #Derive and store t value differences
        t_vals = []
        d_vals = []
        for i in range(3):
            space_d = vn.dot(othr_pos[:,i]-wind_pos.T)
            space_dt = space_d/vm
            d_vals.append(float(space_d))
            t_vals.append(float(space_dt))
       
        
        
        #store time values as numpy array
        t_vals = np.array(t_vals)
        #store distance values as numpy array
        d_vals = np.array(d_vals)
        #get solution of plane velocity using solve_plane
        svna,svn,svm = mtr.solve_plane((othr_pos-wind_pos.T).T,t_vals)

        #Get velocity components (mag x normal vector)
        vc = svn*svm
        
        #check the solutions agree to 1 km/s
        self.assertEqual(round(svm),round(vm))
        
        #get solution of coefficents at wind
        test = np.array(mtr.solve_coeff(wind_pos.T,vn))

        #check I get the same Z value for wind position
        nz = -(test[0]*x+test[1]*y+test[3])/test[2]
        self.assertEqual(int(nz),int(z))

        
        #for far out postion
        nzf = -(test[0]*xf+test[1]*yf+test[3])/test[2]
        self.assertEqual(int(nzf),int(zf))

if __name__=='__main__':
    unittest.main()