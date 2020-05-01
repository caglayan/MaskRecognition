import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    mask_cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=mask_cascade.detectMultiScale(gray,1.01, 4)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    mask1 = cv2.inRange(img, (0, 255, 0), (200, 255,200 ))
    if cv2.countNonZero(mask1) < 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Wear Mask'
            frame = cv2.putText(img, text, (100, 150), font, 2, (10, 10, 100), 2, cv2.LINE_AA)
            cv2.imshow('img', frame)


    # Display the output
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
