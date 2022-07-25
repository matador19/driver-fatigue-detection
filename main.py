#Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2

# Create an instance of TKinter Window or frame
win = Tk()
win.title("DRIVER FATIGUE DETECTION")

# Set the size of the window
win.geometry("600x600")

# Create a Label to capture the Video frames
labelone =Label(win)
labelone.grid(row=0, column=0)
cap= cv2.VideoCapture(0)
fatigue_level=0.71
SleepLabel=Label(win,text=f'Fatigue level:{fatigue_level}',font=("Arial", 30))
SleepLabel.grid(row=1,column=0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   labelone.imgtk = imgtk
   labelone.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   labelone.after(20, show_frames)
def fatiguefunction():
   show_frames()
   warningimage = Image.open("UIAssets/OverlayImage.png")
   resize_image = warningimage.resize((55, 55))
   img = ImageTk.PhotoImage(resize_image)
   warninglabel = Label(win, image=img)
   warninglabel.grid(row=2, column=0)
   win.mainloop()

def detectionmode():
   show_frames()
   win.mainloop()

if fatigue_level>0.7 :
  fatiguefunction()
else:
   detectionmode()