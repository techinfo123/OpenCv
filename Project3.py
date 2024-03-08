# NumberPlate_Detection

import cv2
import numpy as np

widthimage = 480
heightimage = 640
platecascade = cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
minarea =500
color = (255,0,255)
cap = cv2.VideoCapture(0)  # Try different camera indices
cap.set(3, widthimage)
cap.set(4, heightimage)
cap.set(10, 150)
count =0

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberplates = platecascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberplates:
        area =w*h
        if area>minarea:
          cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
          cv2.putText(img,'numberplate',(x,y-5),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
          imgroi = img[y:y+h,x:x+w]
          cv2.imshow('imgroi',imgroi)

    cv2.imshow('result',img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Resources/scanned/noplate_'+str(count)+'.jpg',imgroi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,'Scan Saved',(150,265),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv2.imshow('result',img)
        cv2.waitKey(500)
        count+=1

