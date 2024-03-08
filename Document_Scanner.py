import cv2
import numpy as np

widthimage = 480
heightimage = 640
cap = cv2.VideoCapture(0)  # Try different camera indices
cap.set(3, widthimage)
cap.set(4, heightimage)
cap.set(10, 150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

def contours(img, imgcontour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxarea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxarea and len(approx) == 4:
                biggest = approx
                maxarea = area
    if biggest.any():
        cv2.drawContours(imgcontour, [biggest], -1, (255, 0, 0), 20)
    return biggest

def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))
    mypointsnew = np.zeros((4,1,2),np.int32)
    add = mypoints.sum(1)
    # print('add',add)

    mypointsnew[0]=mypoints[np.argmin(add)]
    mypointsnew[3]=mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]
    # print('newpoints',mypointsnew)
    return mypointsnew

def getWrap(img, biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthimage, 0], [0, heightimage], [widthimage, heightimage]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthimage, heightimage))
    return imgOutput

while True:
    success, img = cap.read()
    if not success:
        continue  # If frame not read successfully, skip to next iteration

    img = cv2.resize(img, (widthimage, heightimage))
    imgcontour = img.copy()
    imgThres = preProcessing(img)
    biggest = contours(imgThres, imgcontour)
    # print(biggest)

    if biggest.any():  # Check if any contour was found
        imagewrapped = getWrap(img, biggest)
        cv2.imshow('result', imagewrapped)
    else:
        cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
