#Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from crop_eyes import combined_cropped_eyes_image
from neural_network import model_eval

# Create an instance of TKinter Window or frame
win = Tk()
win.title("DRIVER FATIGUE DETECTION")

# Set the size of the window
win.geometry("600x600")

# Create a Label to capture the Video frames
labelone = Label(win)
labelone.grid(row=0, column=0)
cap = cv2.VideoCapture(1)
fatigue_level = 0
fatigue_label = Label(win, text=f'Fatigue level:{fatigue_level}', font=("Arial", 30))
fatigue_label.grid(row=1, column=0)


# Define function to show frame
def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)

    # Evaluate the frame
    predicted_class = eval_frame(cv2image)

    if predicted_class is None:
        predicted_class = "No face"

    fatigue_label.configure(text=predicted_class)

    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)

    labelone.imgtk = imgtk
    labelone.configure(image=imgtk)

    # Convert image to PhotoImage
    # eyes_img = ImageTk.PhotoImage(image=cropped_img)

    # Repeat after an interval to capture continiously
    labelone.after(20, show_frames)


def eval_frame(img):
    cropped = combined_cropped_eyes_image(img)

    if cropped is None:
        return

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return model_eval(cropped)


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


if fatigue_level > 0.7 :
    fatiguefunction()
else:
    detectionmode()
