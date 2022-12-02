import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import asksaveasfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import cv2
import numpy as np
from object_size import pixel_to_millimeters


# --------- FUNCTION DEFINITIONS -------------
def alignImages(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.50

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imshow('matches',imMatches)
    # cv2.waitKey()

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


# --------- FUNCTION DEFINITIONS END -------------

class displacementMeasurement(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #Defining camera parameters:
        self.GoPro = 3
        self.cap = cv2.VideoCapture(self.GoPro)
        while not self.cap.isOpened():
            print("Error: Camera not found")
            raise IOError("Error: Cannot open webcam")


        #Configure root tk class
        self.master = master
        self.master.title("Displacement Measurement Using CV")
        self.master.iconbitmap("assets/Voltage - Continuous Input.ico")
        self.master.geometry("1500x800")

        self.create_widgets()
        self.pack()
        self.run = False

        # Defining Color intervals to recognize
        
        self.ORANGE_MIN = np.array([5, 50, 50],np.uint8)
        self.ORANGE_MAX = np.array([15, 255, 255],np.uint8)

        self.BLACK_MIN = np.array([0, 0, 0],np.uint8)
        self.BLACK_MAX = np.array([140, 140, 140],np.uint8)

        self.GREEN_MIN = np.array([40, 40,40],np.uint8)
        self.GREEN_MAX = np.array([70, 255,255],np.uint8)


        # Defining the Coordinate Vectors for later analysis
        self.X, self.Y, self.X_avg, self.Y_avg, self.T = [],[],[],[],[]

        # For Conversion
        self.refHeight = 8.75
        self.refWidth = 47.8
        self.ratioPXLtoMM = 1

    def create_widgets(self):
        #The main frame is made up of three subframes
        self.HASELSettingsFrame = HASELSettings(self, title ="HASEL Properties")
        self.HASELSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(10,0), padx=(20,20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=2, column=1, pady=(10,0), padx=(20,20), ipady=10)

        self.calibrationSettingsFrame = calibrationSettings(self, title="Calibration Settings")
        self.calibrationSettingsFrame.grid(row=1, column=1, pady=(10,0), padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=3, column=2, pady=(20,0), ipady=10)

    def startPreview(self):

        self.continueRunning = True

        self.inputSettingsFrame.startButton['state'] = 'disabled'
        self.calibrationSettingsFrame.previewStartButton['state'] = 'disabled'
        self.inputSettingsFrame.stopButton['state'] = 'disabled'


        self.master.after(10, self.runPreview)

    def runPreview(self):
        

        _, self.frame = self.cap.read()
        
        self.blurred_frame = cv2.GaussianBlur(self.frame, (5, 5), 0)
        self.hsv = cv2.cvtColor(self.blurred_frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.GREEN_MIN, self.GREEN_MAX)
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in self.contours:
            self.area = cv2.contourArea(contour)
            if self.area > 8000: #Change the area size accordingly, This is only to reduce any noise
                cv2.drawContours(self.frame, contour, -1, (0, 255, 0), 3)

        cv2.imshow('MASK PREVIEW: PRESS ANY KEY TO EXIT', self.mask)

        self.key = cv2.waitKey(1)
        if self.key != -1: 
            self.master.after(10, self.closePreview)
            return
            
        if self.continueRunning == True:
            self.master.after(10, self.runPreview)

    def closePreview(self):
        
        self.continueRunning = False
        cv2.destroyWindow('MASK PREVIEW: PRESS ANY KEY TO EXIT')
        # self.cap.release()
        self.inputSettingsFrame.startButton['state'] = 'enabled'
        self.inputSettingsFrame.stopButton['state'] = 'enabled'
        self.calibrationSettingsFrame.previewStartButton['state'] = 'enabled'

    def stopPreview(self):
        self.continueRunning = False

    def startTask(self):
        #Prevent user from starting task a second time
        self.inputSettingsFrame.startButton['state'] = 'disabled'

        # Explanation of self.countRun:
        # self.countRun = 0: It should only be 0 once! Thats when the reference is obtained
        # self.countRun = 1: If it's 1, then we get a new pixelToMM ration (every so often)
        # self.countRun in range(1,100): Just to let time pass until countRun = 1
        self.countRun = 0 

        #Shared flag to alert task if it should stop
        self.continueRunning = True
        #time.sleep(0.5) #Sometimes it takes some time to open the webcam


        # Defining and clearing all relevant parameters for a new displacement measurement
        self.X, self.Y, self.X_avg, self.Y_avg, self.T, self.X_ref,  self.Y_ref, self.x_ref_avg, self.y_ref_avg = [],[],[],[],[], [],[], [],[]
        self.positiveDisplacement = 0
        self.negativeDisplacement = 0
        self.maxDisplacement = 0
        self.time = 0
        self.frame_ref = cv2.imread("img/reference.png")
        
        #spin off call to check 
        self.master.after(10, self.runTask)

    def runTask(self):
        
        _, self.frame = self.cap.read()
        
        # Ilyas Implementation: #Out commented for now this can be done later
        # self.frame, _ = alignImages(self.frame, self.frame_ref)

        self.blurred_frame = cv2.GaussianBlur(self.frame, (5, 5), 0)
        self.hsv = cv2.cvtColor(self.blurred_frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.GREEN_MIN, self.GREEN_MAX)
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.count = 0
        #print("-----------------------------------------")

        self.countContours = 0 # Only for making sure one contour is detected
        for contour in self.contours:

            self.area = cv2.contourArea(contour)
            self.countContours += 1

            if self.countContours > 1: # If more than one contour, skip
                break 
            
            if self.area > 8000: #Change the area size accordingly, This is only to reduce any noise
                self.X = [] #Clear for new contour i.e. new frame
                self.Y = []
                self.dimension = contour.shape

                # cv2.drawContours(self.frame, contour, -1, (0, 255, 0), 3)

                if self.countRun == 0:
                    for i in range(0, self.dimension[0]):
                        self.X_ref.append(contour[i][0][0])
                        self.Y_ref.append(contour[i][0][1])
                        self.x_ref_avg = sum(self.X_ref)/len(self.X_ref)
                        self.y_ref_avg = sum(self.Y_ref)/len(self.Y_ref)

                elif self.countRun == 1:
                    self.ratioPXLtoMM = pixel_to_millimeters(contour, float(self.calibrationSettingsFrame.widthEntry.get()), float(self.calibrationSettingsFrame.heightEntry.get()))
                    print("NEW RATIO:", self.ratioPXLtoMM)


                for i in range(0, self.dimension[0]):
                    self.X.append(contour[i][0][0])
                    self.Y.append(contour[i][0][1])
                
                # Capturing relevant data for plot
                assert len(self.X) != 0,"Division by 0 X Vector"
                assert len(self.Y) != 0,"Division by 0 Y Vector"

                # Takinf the average of the whole contour for error reduction
                self.x_avg = sum(self.X)/len(self.X)
                self.y_avg = sum(self.Y)/len(self.Y)
                print("GERADE: " ,self.y_avg, "  REFERENCE:", self.y_ref_avg)
                self.X_avg.append((self.x_avg - self.x_ref_avg)/self.ratioPXLtoMM) #This gives you the actual displacement compared to the reference
                self.Y_avg.append((self.y_avg - self.y_ref_avg)/self.ratioPXLtoMM)
                print("MM Änderung: " ,(self.y_avg - self.y_ref_avg)/self.ratioPXLtoMM)
                self.T.append(self.time)
                self.time += 1
                

                self.countRun += 1
                if self.countRun == 100:
                    self.countRun = 1


        #cv2.imshow('HASEL', self.mask)

        self.graphDataFrame.ax.cla()
        self.graphDataFrame.ax.set_title("Strain of HASEL at Load: " + self.inputSettingsFrame.loadEntry.get() + " g")
        self.graphDataFrame.ax.plot(self.Y_avg)
        self.graphDataFrame.graph.draw()
        
        if self.continueRunning:
            self.master.after(10, self.runTask)
        
        else:
            # self.cap.release()

            self.positiveDisplacement = max(self.Y_avg)
            self.negativeDisplacement = min(self.Y_avg)
            self.maxDisplacement = max(abs(self.positiveDisplacement), abs(self.negativeDisplacement))
            print(self.maxDisplacement)

            cv2.destroyAllWindows
            print("Stopped Running")
            self.inputSettingsFrame.startButton['state'] = 'enabled'
              
    def stopTask(self):
        #call back for the "stop task" button
        self.continueRunning = False

        #Plottinh the result

        plt.plot(self.T, self.Y_avg) 
        # naming the x axis 
        plt.xlabel('Time') 
        # naming the y axis 
        #plt.ylabel('Displacement [mm]') 
        # giving a title to my graph 
        plt.text(-5, 60, "Max Displ: " + str(self.maxDisplacement), fontsize = 22)
        plt.text(-30, 60, "Load [g]: " + str(self.inputSettingsFrame.loadEntry.get()), fontsize = 22)
        

        plt.title(self.HASELSettingsFrame.HASELNumberEntry.get() + "::" +  self.HASELSettingsFrame.optionVar.get() + " | " + self.inputSettingsFrame.loadEntry.get() + " g | " + self.inputSettingsFrame.actuationVoltageEntry.get() + " V") 
            
        # function to show the plot 
        plt.show()       


    def exportFigure(self):
        #self.f = asksaveasfile(initialfile = self.HASELSettingsFrame.HASELNumberEntry.get() + ".png",
        #defaultextension=".png",filetypes=[("All Files","*.*"),("PNG","*.png")], initialdir='/Users/clemenschristoph/GitHub/hasel-test-rig/HASEL_DATA',title='Export' + self.HASELSettingsFrame.HASELNumberEntry.get())
        #print(self.f)
        #if self.f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            #return
            
        plt.savefig("HASEL_DATA/" + self.HASELSettingsFrame.HASELNumberEntry.get()+ "_" + self.inputSettingsFrame.loadEntry.get() + "_" + self.inputSettingsFrame.actuationVoltageEntry.get())
        plt.close()

        with open('HASEL_DATA/' + self.HASELSettingsFrame.HASELNumberEntry.get() + "_" + self.inputSettingsFrame.loadEntry.get() + "_" + self.inputSettingsFrame.actuationVoltageEntry.get(), 'w') as self.f:
            self.f.write("HASEL Number: " + self.HASELSettingsFrame.HASELNumberEntry.get() + "\n")
            self.f.write("HASEL Type: " + self.HASELSettingsFrame.optionVar.get() + "\n")
            self.f.write("Comments: " + self.HASELSettingsFrame.HASELCommentEntry.get() + "\n")
            self.f.write("Load [g]: " + self.inputSettingsFrame.loadEntry.get() + "\n")
            self.f.write("Actuation Voltage [V]: " + self.inputSettingsFrame.actuationVoltageEntry.get() + "\n")
            self.f.write("Max Displacement: [mm] " + str(self.maxDisplacement) + "\n")
            self.f.write("\n")
            for i in range(0, len(self.Y_avg)):
                self.f.write(str(self.Y_avg[i]) + " ")
            self.f.write("\n")

    def exportTXT(self):
        with open('HASEL_DATA/' + self.HASELSettingsFrame.HASELNumberEntry.get() + "_" + self.inputSettingsFrame.loadEntry.get, 'w') as self.f:
            self.f.write("HASEL Number: " + self.HASELSettingsFrame.HASELNumberEntry.get() + "\n")
            self.f.write("HASEL Type: " + self.HASELSettingsFrame.optionVar.get() + "\n")
            self.f.write("Comments: " + self.HASELSettingsFrame.HASELCommentEntry.get() + "\n")
            self.f.write("Load [g]: " + self.inputSettingsFrame.loadEntry.get() + "\n")
            self.f.write("Actuation Voltage [V]: " + self.inputSettingsFrame.actuationVoltageEntry.get() + "\n")
            self.f.write("Max Displacement: [mm] " + str(self.maxDisplacement) + "\n")
            self.f.write("\n")
            for i in range(0, len(self.Y_avg)):
                self.f.write(str(self.Y_avg[i]) + " ")
            self.f.write("\n")
        


    
class HASELSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.HASELTypes = ["HASEL_Orginial", "HASEL_Small", "Froggy", "Froggy_Half", "Butterfly", "Buggy"]
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

        self.HASELCommentLabel = ttk.Label(self, text="Comments")
        self.HASELCommentLabel.grid(row=4,sticky='w', padx=self.xPadding, pady=(10,0))

        self.HASELCommentEntry = ttk.Entry(self)
        self.HASELCommentEntry.insert(0, "")
        self.HASELCommentEntry.grid(row=5, sticky="ew", padx=self.xPadding)    

class calibrationSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent

        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30,30)
        self.create_widgets()


    def create_widgets(self):

        self.widthLabel = ttk.Label(self, text="Ref Width [mm]")
        self.widthLabel.grid(row=0, column= 0, sticky='w', padx=self.xPadding, pady=(10,0))

        self.widthEntry = ttk.Entry(self)
        self.widthEntry.insert(0, "40.7")
        self.widthEntry.grid(row=1, column= 0, sticky="ew", padx=self.xPadding)

        self.heightLabel = ttk.Label(self, text="Ref Height [mm]")
        self.heightLabel.grid(row=2, column= 0, sticky='w', padx=self.xPadding, pady=(10,0))

        self.heightEntry = ttk.Entry(self)
        self.heightEntry.insert(0, "8.75")
        self.heightEntry.grid(row=3, column= 0, sticky="ew", padx=self.xPadding)

        self.previewStartButton = ttk.Button(self, text="Open Preview", command=self.parent.startPreview)
        self.previewStartButton.grid(row=4, column=0, sticky='w', padx=self.xPadding, pady=(10,0))
        
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

        self.exportButton = ttk.Button(self, text="Export Figure", command=self.parent.exportFigure)
        self.exportButton.grid(row=5, column=0, sticky='w', padx=self.xPadding, pady=(10,0))

        #self.stopButton = ttk.Button(self, text="Export txt File", command=self.parent.exportTXT)
        #self.stopButton.grid(row=5, column=1, sticky='e', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(7,5), dpi=150)
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