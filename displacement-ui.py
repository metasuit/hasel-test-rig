import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import cv2
import numpy as np



class displacementMeasurement(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #Configure root tk class
        self.master = master
        self.master.title("Displacement Measurement Using CV")
        self.master.iconbitmap("assets/Voltage - Continuous Input.ico")
        self.master.geometry("1100x600")

        self.create_widgets()
        self.pack()
        self.run = False

        # Defining Color intervals to recognize
        
        self.ORANGE_MIN = np.array([5, 50, 50],np.uint8)
        self.ORANGE_MAX = np.array([15, 255, 255],np.uint8)

        # Defining the Coordinate Vectors for later analysis
        self.X, self.Y, self.X_avg, self.Y_avg, self.T = [],[],[],[],[]
        self.time = 0
        self.cap = cv2.VideoCapture(3)
        time.sleep(0.5)

    def create_widgets(self):
        #The main frame is made up of three subframes
        self.HASELSettingsFrame = HASELSettings(self, title ="HASEL Properties")
        self.HASELSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20,0), ipady=10)


    def startTask(self):
        #Prevent user from starting task a second time
        self.inputSettingsFrame.startButton['state'] = 'disabled'

        #Shared flag to alert task if it should stop
        self.continueRunning = True
        self.cap = cv2.VideoCapture(3)
        self.X, self.Y, self.X_avg, self.Y_avg, self.T = [],[],[],[],[]
        self.time = 0
        
        #spin off call to check 
        self.master.after(500, self.runTask)

    def runTask(self):
        
        _, self.frame = self.cap.read()

        # Ilyas Implementation:
        # frame = align_image(frame)

        self.blurred_frame = cv2.GaussianBlur(self.frame, (5, 5), 0)
        self.hsv = cv2.cvtColor(self.blurred_frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.ORANGE_MIN, self.ORANGE_MAX)
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.count = 0
        #print("-----------------------------------------")
        for contour in self.contours:
            self.area = cv2.contourArea(contour)
            self.count += 1
            
            if self.area > 8000:
                self.X = []
                self.Y = []
                cv2.drawContours(self.frame, contour, -1, (0, 255, 0), 3)
                self.dimension = contour.shape
                for i in range(0, self.dimension[0]):
                    self.X.append(contour[i][0][0])
                    self.Y.append(contour[i][0][1])
            #else:
                #print("Error: No contours or too many contours detected")

        # Capturing relevant data for plot
        if len(self.X) != 0 and len(self.Y) != 0:
            self.x_avg = sum(self.X)/len(self.X)
            self.y_avg = sum(self.Y)/len(self.Y)
            self.X_avg.append(self.x_avg)
            self.Y_avg.append(self.y_avg)
            self.T.append(self.time)
            self.time += 1

        #cv2.imshow('HASEL', self.frame)
        self.graphDataFrame.ax.cla()
        self.graphDataFrame.ax.set_title("Acquired Data")
        self.graphDataFrame.ax.plot(self.Y_avg)
        self.graphDataFrame.graph.draw()
        
        if self.continueRunning:
            self.master.after(10, self.runTask)
        
        else:
            self.cap.release()
            cv2.destroyAllWindows
            print("Stopped Running")
            self.inputSettingsFrame.startButton['state'] = 'enabled'
            
   
    def stopTask(self):
        #call back for the "stop task" button
        self.continueRunning = False
        
        
        

class HASELSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.HASELTypes = ["HASEL_Orginial", "HASEL_Small", "Froggy", "Froggy_Half"]
        self.optionVar = tk.StringVar(self)

        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30,30)
        self.create_widgets()

    def create_widgets(self):

        self.HASELNumberLabel = ttk.Label(self, text="Hasel Number")
        self.HASELNumberLabel.grid(row=0,sticky='w', padx=self.xPadding, pady=(10,0))

        self.HASELNumberEntry = ttk.Entry(self)
        self.HASELNumberEntry.insert(0, "H001")
        self.HASELNumberEntry.grid(row=1, sticky="ew", padx=self.xPadding)

        self.HASELTypeLabel = ttk.Label(self, text="Hasel Type")
        self.HASELTypeLabel.grid(row=2,sticky='w', padx=self.xPadding, pady=(10,0))


        self.HASELTypeMenu = ttk.OptionMenu(
            self,
            self.optionVar,
            self.HASELTypes[0],
            *self.HASELTypes)
        self.HASELTypeMenu.grid(row=3, sticky="ew", padx=self.xPadding)       
        

class inputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (30,30)
        self.create_widgets()

    def create_widgets(self):
        self.loadLabel = ttk.Label(self, text="Load [g]")
        self.loadLabel.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.loadEntry = ttk.Entry(self)
        self.loadEntry.insert(0, "100")
        self.loadEntry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.actuationVoltageLabel = ttk.Label(self, text="Actuation Voltage [V]")
        self.actuationVoltageLabel.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.actuationVoltageEntry = ttk.Entry(self)
        self.actuationVoltageEntry.insert(0, "6000")
        self.actuationVoltageEntry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.startButton = ttk.Button(self, text="Start Task", command=self.parent.startTask)
        self.startButton.grid(row=4, column=0, sticky='w', padx=self.xPadding, pady=(10,0))

        self.stopButton = ttk.Button(self, text="Stop Task", command=self.parent.stopTask)
        self.stopButton.grid(row=4, column=1, sticky='e', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(7,5), dpi=100)
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_title("Time Dependent Strain of HASEL")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

#Creates the tk class and primary application "voltageContinuousInput"
root = tk.Tk()
app = displacementMeasurement(root)

#start the application
app.mainloop()