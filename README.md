# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV window.

### Program:
```

import cv2
import matplotlib.pyplot as plt
import numpy as np
img1 = cv2.imread('image_01.png', 0)
img2 = cv2.imread('image_02.png', 0)
img3 = cv2.imread('image_03.png', 0)
plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(img1 , cmap='gray');plt.title("image 1");plt.axis("ON")
plt.subplot(142)
plt.imshow(img2 , cmap='gray');plt.title("image 2");plt.axis("ON")
plt.subplot(143)
plt.imshow(img3 , cmap='gray');plt.title("image 3");plt.axis("ON")
plt.figure(figsize=(20,20))
plt.subplot(141);plt.imshow(cv2.resize(img1,(1000,1000)),cmap='grey');plt.title("without glass");plt.axis("on")
plt.subplot(142);plt.imshow(cv2.resize(img2,(1000,1000)),cmap='grey');plt.title("with glass");plt.axis("on")
plt.subplot(143);plt.imshow(cv2.resize(img3,(1000,1000)),cmap='grey');plt.title("group");plt.axis("on")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect_face(img):
    face_img=img.copy()
    face_rect=face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(127,0,255),10)
        cv2.imshow('FACE DETECTION',face_img)
    return face_img
res_img1=detect_face(img1)
res_img2=detect_face(img2)
res_img3=detect_face(img3)
plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(res_img1, cmap='gray');plt.title("image 1");plt.axis("ON")


plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(res_img2, cmap='gray');plt.title("image 1");plt.axis("ON")
plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(res_img3, cmap='gray');plt.title("image 1");plt.axis("ON")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
def detect_eye(img):
    eye_img=img.copy()
    eye_rect=eye_cascade.detectMultiScale(eye_img)
    for(x,y,w,h) in eye_rect:
        cv2.rectangle(eye_img,(x,y),(x+w,y+h),(127,0,255),10)
        cv2.imshow('FACE DETECTION',eye_img)
    return eye_img
re_img1=detect_eye(img1)
re_img2=detect_eye(img2)
re_img3=detect_eye(img3)
plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(re_img1, cmap='gray');plt.title("image 1");plt.axis("ON")
plt.figure(figsize=(20,12))
plt.subplot(142)
plt.imshow(re_img2, cmap='gray');plt.title("image 1");plt.axis("ON")
plt.figure(figsize=(20,12))
plt.subplot(143)
plt.imshow(re_img3, cmap='gray');plt.title("image 1");plt.axis("ON")
cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()
ret, frame = cap.read(0)
frame = detect_face(frame)
im=ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
plt.title('video capture')

while True:
    ret,frame=cap.read(0)
    frame= detect_face(frame)
    im.set_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.pause(0.10)
cap.release()
plt.close()
```
### Output:
<img width="1201" height="403" alt="download" src="https://github.com/user-attachments/assets/770b55b9-ce10-4484-86e4-afe37d301077" />

### face detection
<img width="401" height="510" alt="download" src="https://github.com/user-attachments/assets/ffdac9a3-215b-4a4b-a3d1-1cb09300a784" />

<img width="393" height="389" alt="download" src="https://github.com/user-attachments/assets/f0cced9f-963b-4629-aa9d-d43680ba228d" />

<img width="411" height="273" alt="download" src="https://github.com/user-attachments/assets/569dbe22-947c-4a23-a4c4-2f6bc4186270" />

### eye detection
<img width="401" height="510" alt="download" src="https://github.com/user-attachments/assets/74a01636-2b0c-4518-bd83-aad77663a725" />

<img width="393" height="389" alt="download" src="https://github.com/user-attachments/assets/b7bc1027-2c08-43e8-971f-a4da3a4206a7" />

<img width="411" height="273" alt="download" src="https://github.com/user-attachments/assets/a3beb2c8-d906-4c67-9091-a481721ea0b5" />

### Result:
thus the program has been executed successfully.
