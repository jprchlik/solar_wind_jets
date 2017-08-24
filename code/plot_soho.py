import matplotlib as mpl
mpl.use('TkAgg',warn=False,force=True)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 24



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import pandas as pd 
from fancy_plot import fancy_plot
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import sys

#check the python version to use one Tkinter syntax or another
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


#main gui class
class gui_c(Tk.Frame):

    #init gui class
    def __init__(self,parent):
        #create Tk frame
        Tk.Frame.__init__(self,parent,background='white')

        #create class parent variable
        self.parent = parent

        #get center of screen
        self.centerWindow()
        self.FigureWindow()
        #create soho table
        self.get_soho_files()
        #min value in day list
        self.minday = self.soho_df.index.min()
        #set the current day to the mininum day
        self.curday = self.soho_df.index.min()
        #create string varaible for day
        self.strday = str(self.curday)
        #get maximum value in list
        self.maxday = self.soho_df.index.max()

        #init the user interface
        self.initUI()
#Start the creation of the window and GUI
        #make inital plot
        self.make_plot()



#Create window in center of screen
    def centerWindow(self):
        w = 2000
        h = 1200
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw-w)/2
        y = (sh-h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w,h,x,y))



    #get all soho files
    def get_soho_files(self):

        #find all soho files in data directory
        f_soho = glob('../soho/data/*txt')
        
        #read in all soho files in data directory
        df_file = (pd.read_table(f,skiprows=28,engine='python',delim_whitespace=True) for f in f_soho)
        
        #create one large array with all soho information in range
        soho_df = pd.concat(df_file,ignore_index=True)
        
        #convert columns to datetime column
        soho_df['time_dt'] = pd.to_datetime('20'+soho_df['YY'].astype('str')+':'+soho_df['DOY:HH:MM:SS'],format='%Y:%j:%H:%M:%S')
        
        #set index to be time
        soho_df.set_index(soho_df['time_dt'],inplace=True)

        #make table a class variable
        self.soho_df = soho_df

    #set current limit
    def set_day_limit(self):
        pad = timedelta(hours=14)
        self.ax[0].set_xlim([self.curday-pad,self.curday+pad])
        for tick in self.ax[2].get_xticklabels(): tick.set_rotation(25)
        self.canvas.draw()
   
    #create window for figure and buttons
    def FigureWindow(self):
        #set the information based on screen size
        x =  self.parent.winfo_screenwidth()
        y =  self.parent.winfo_screenheight()

        #ploting frame
        pframe = Tk.Frame(self)

        #aspect ratio of given screen
        aratio = float(x)/float(y)

 
        #create a figure and plot
        self.fig, self.ax = plt.subplots(nrows=3,figsize=(6*aratio,6*aratio),sharex=True)
        self.fig.subplots_adjust(hspace=0.0001,wspace=0.0001)

        #Create window for the plot
        self.canvas = FigureCanvasTkAgg(self.fig,master=self)
        #Draw the plot
        self.canvas.draw()
        #Turn on matplotlib widgets
        self.canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
        #Display matplotlib widgets
        self.toolbar = NavigationToolbar2TkAgg(self.canvas,self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
        #Connect mpl to mouse clicking
        #self.fig.canvas.mpl_connect('button_press_event',self.on_click_event)


        #create button to go down an order
        downbutton = Tk.Button(master=pframe,text='Decrease One Day',command=self.decreaseday)
        downbutton.pack(side=Tk.LEFT)
        #create button to go up an order
        upbutton = Tk.Button(master=pframe,text='Increase One Day',command=self.increaseday)
        upbutton.pack(side=Tk.LEFT)

        pframe.pack(side=Tk.TOP)

    #plot soho wind data
    def make_plot(self):
        #plot solar wind speed
        self.ax[0].scatter(self.soho_df.index,self.soho_df.SPEED,color='black')
        self.ax[0].set_ylabel('$|\mathrm{V}|$ [km/s]')
        fancy_plot(self.ax[0])
        
        #plot solar wind density
        self.ax[1].scatter(self.soho_df.index,self.soho_df.Np,color='black')
        self.ax[1].set_ylabel('n$_\mathrm{p}$ [cm$^{-3}$]')
        self.ax[1].set_yscale('log')
        fancy_plot(self.ax[1])
        
        #Thermal Speed
        self.ax[2].scatter(self.soho_df.index,self.soho_df.Vth,color='black')
        self.ax[2].set_ylabel('w$_\mathrm{p}$ [km/s]')
        self.ax[2].set_xlabel('Time')
        fancy_plot(self.ax[2])

        #set plotting limit
        self.set_day_limit()


 


#Command to increase the order to plot new aia image
    def increaseday(self):
        self.curday = self.curday+timedelta(hours=24)
        if self.curday > self.maxday:
            self.curday = self.minday
        self.strday.set(str(self.curday))
        self.set_day_limit()

#Command to decrease order to plot new aia image
    def decreaseday(self):
        self.curday = self.curday-timedelta(hours=24)
        if self.curday < self.minday:
            self.curday = self.maxday
        self.strday.set(str(self.curday))
        self.set_day_limit()




#Tells Why Order information is incorrect
    def onError(self):
        if self.error == 1:
            box.showerror("Error","File Not Found")
        if self.error == 4:
            box.showerror("Error","Value Must be an Integer")
        if self.error == 6:
            box.showerror("Error","File is not in Fits Format")
        if self.error == 10:
            box.showerror("Error","Value Must be Float")
        if self.error == 20:
            box.showerror("Error","Must Select Inside Plot Bounds")

#Exits the program
    def onExit(self):
       plt.clf()
       plt.close()
       self.quit()
       self.parent.destroy()

#Intialize the GUI
    def initUI(self):
#set up the title 
        self.parent.title("Find SOHO Discontinuities")
#create frame for plotting
        frame = Tk.Frame(self,relief=Tk.RAISED,borderwidth=1)
        frame.pack(fill=Tk.BOTH,expand=1)
        self.pack(fill=Tk.BOTH,expand=1)

#set up okay and quit buttons
        quitButton = Tk.Button(self,text="Quit",command=self.onExit)
        quitButton.pack(side=Tk.RIGHT,padx=5,pady=5)


#set up day box
        dayText = Tk.StringVar()
        dayText.set("Plot Time")
        dayDir = Tk.Label(self,textvariable=dayText,height=4)
        dayDir.pack(side=Tk.LEFT)
#Add so day can be updated
        self.strday = Tk.StringVar()
        self.strday.set(str(self.curday))
        self.dayval = Tk.Entry(self,textvariable=self.strday,width=25)
        self.dayval.bind("<Return>",self.on_day_box)
        self.dayval.pack(side=Tk.LEFT,padx=5,pady=5)

#set up Submenu
        menubar = Tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Tk.Menu(menubar)
        subMenu = Tk.Menu(fileMenu)
#create another item in menu
        fileMenu.add_separator()

        fileMenu.add_command(label='Exit',underline=0,command=self.onExit)

#Function for retrieving order from text box
    def on_day_box(self,event):
#release cursor from entry box and back to the figure
#needs to be done otherwise key strokes will not work
        self.fig.canvas._tkcanvas.focus_set()
        m = 0
        while m == 0:
            try:
                day = self.dayval.get()
                day = pd.to_datetime(day)
                if ((day >= self.minday) & (day <= self.maxday)):
                    m = 1
                    self.curday = day
                    self.set_day_limit()
                else:
#Error day is out of range
                    self.error = 3
                    self.onError()
            except ValueError:
#Error day is not an integer
                self.error = 4
                self.onError()




#main loop
def main():
    global root
    root = Tk.Tk()
    app = gui_c(root)
    root.mainloop()


if __name__=="__main__":
#create root frame
   main()



