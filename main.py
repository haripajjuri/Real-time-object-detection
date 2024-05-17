import tkinter as tk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk 
import numpy as np

width, height = 1200, 600

vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 

app = tk.Tk()
app.geometry("1000x700")

#frame = tk.Frame(app, width=900,height=500,background="green")
#frame.pack()

canvas = tk.Canvas(app,width=1150,height=550,background="gray51")
canvas.pack()

#label_widget = tk.Label(app, width=120, height=30, background="green")
#label_widget.pack()


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

model = YOLO("best.pt")
list = []

def open_camera(): 
    global image_id
    _, frame = vid.read() 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    captured_image = Image.fromarray(img) 
    photo = ImageTk.PhotoImage(image=captured_image) 
    #label_widget.photo_image = photo_image 
    #canvas.create_image((100,100),image = photo_image)
    canvas.photo = photo

    if image_id:
        canvas.itemconfig(image_id,image = photo)
    else:
        image_id = canvas.create_image((0,0), image=photo, anchor='nw')
        canvas.configure(width=photo.width(), height=photo.height())
    canvas.after(5,open_camera)



image_id = None

br = tk.Label(app, text="\n")

br.pack()  

button1 = tk.Button(app, text="Open Camera", command= open_camera)
button1.pack() 


app.mainloop() 