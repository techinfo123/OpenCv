import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,480)
cap.set(10,100)

colors = [[169,52,0,179,255,255],
        [67,56,157,133,255,255],
        [18,32,214,179,168,247]]

myColorsVal =[[0,127,255],   # GBR
              [0,255,0],
              [128,0,255]]

myPoints = []  # x,y,colorid

def findcolor(img,colors,myColorVal):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newpoints=[]
    for color in colors:
      lower = np.array(color[0:3])
      upper = np.array(color[3:6])
      mask = cv2.inRange(imgHSV, lower, upper)
      x,y = contours(mask)
      cv2.circle(imgResult,(x,y),10,myColorsVal[count],cv2.FILLED)
      if x!=0 and y!=0:
          newpoints.append([x,y,count])
      count+=1
      # cv2.imshow(str(color[0]),mask)
    return newpoints
def contours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h=0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
          # cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)
          peri = cv2.arcLength(cnt,True)
          approx = cv2.approxPolyDP(cnt,0.02*peri,True)
          x,y,w,h =cv2.boundingRect(approx)
    return x+w//2,y

def drawoncanvas(mypoints,myColorvals):
    for points in mypoints:
        cv2.circle(imgResult, (points[0], points[1]), 10, myColorsVal[points[2]], cv2.FILLED)


while True:
    Success,img = cap.read()
    imgResult = img.copy()
    newpoints = findcolor(img, colors , myColorsVal)
    if len(newpoints)!=0:
        for newP in newpoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawoncanvas(myPoints,myColorsVal)
    cv2.imshow('video',imgResult)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break