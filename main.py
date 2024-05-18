#importing packages
import tkinter as tk
from tkinter import ttk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk 
import numpy as np


#initialising camera
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1200) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) 


#initialising a tkinter window and geometry
app = tk.Tk()
app.title("real-time object detection")
app.geometry("1100x700")
styles = ttk.Style(app)
styles.theme_use('classic')

#creating a canvas widget for displaying video output
canvas = tk.Canvas(app,width=1150,height=550,background="gray51")
canvas.pack()


#classnames that training dataset consists
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))
list = []

#loading the dl YOLO model
model = YOLO("model/best.pt")


#function to start object detection
def open_camera():
    global vid
    global temp
    ret, frame = vid.read() 
    if ret:
        try:
            global image_id
            results = model(frame, stream=True)
            for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(box.cls[0])], 3)
                        cls = int(box.cls[0])
                        list.append(classNames[cls])
                        # object details
                        org = [x1+10, y1-20]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.8
                        color = (255, 255, 255)
                        thickness = 2
                        cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
                        
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            captured_image = Image.fromarray(img)
            photo = ImageTk.PhotoImage(image=captured_image) 
            canvas.photo = photo
            if image_id:
                canvas.itemconfig(image_id,image = photo)
            else:
                image_id = canvas.create_image((0,0), image=photo, anchor='nw')
                canvas.configure(width=photo.width(), height=photo.height())
            canvas.after(5,open_camera)
        except Exception as err:
            print(err)
    else:
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1200) 
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) 
        open_camera

#function to stop object detection
def close_camera():
    global temp
    temp = 0
    global image_id
    image_id = None
    canvas.delete("all")
    vid.release()


image_id = None


br = tk.Label(app, text="\n")
br.pack()  

frame = tk.Frame(app, width=400,height=40)
frame.pack()

#start and stop buttons 
button1 = tk.Button(frame, text="Open Camera", command= open_camera)
button1.pack(side="left",padx=10) 

button2 = tk.Button(frame, text="Close Camera", command= close_camera)
button2.pack(side="right",padx=10) 

app.mainloop() 