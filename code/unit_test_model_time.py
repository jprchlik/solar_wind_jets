#start unit testing
import unittest
import model_time_range as mtr
import numpy as np

#Test to make sure the plane solution works 2018/04/24 J. Prchlik
class check_solution(unittest.TestCase):
    
    def test_plane_creation(self):
        #equation for plane 
        self.a,self.b,self.c,self.d = 1000.0,100.0,100.0,100000.0 #km
        #set wind position 
        self.x = 200.
        self.y = 0.
        self.z = -(self.a*self.x+self.b*self.y+self.d)/self.c
        self.wind_pos =  np.matrix([self.x,self.y,self.z])
        self.assertEqual(self.x*self.a+self.y*self.b+self.z*self.c+self.d,0)


    def test_velocity_mag(self):
        a,b,c,d = 1000.0,100.0,100.0,100000.0 #km
        #set wind position 
        x = 200.
        y = 0.
        z = -(a*x+b*y+d)/c
        wind_pos =  np.matrix([x,y,z])
        #magnitude of velocity
        vm = 600. #km/s
        #get normal vector  
        p = np.matrix([a,b,c])
        vn = p/np.linalg.norm(p)

        #create a matrix of space craft posittons
        othr_pos = np.matrix([[500.,200.,700.],  #x values
                             [-200.,400.,-1700.],#y values
                             [1500,-600,40]])    #z values


        #get magnitude of distance from plane measured at wind to other spacecraft
        #Derive and store t value differences
        t_vals = []
        d_vals = []
        for i in range(3):
            #space_d = np.linalg.norm((othr_pos[:,i].T-wind_pos).dot(vn.T))
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
        
        #check the solutions agree to 1 km/s
        self.assertEqual(round(svm),round(vm))
        
        #get solution of coefficents at wind
        print(mtr.solve_coeff(wind_pos.T,svn))

        #for j,i in enumerate(d_vals):
        #    print(solve_coeff(i,othr_pos[:,j],svn))
if __name__=='__main__':
    unittest.main()