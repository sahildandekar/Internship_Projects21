'''
Description: This is a subcode for the main files and there we are going to define function used to perform operation on images
Author: Sahil Dandekar
Project no: 01

'''
import cv2
import numpy as np

'''
Function for convert image to HSV form
    Arguments : 1. Image input
'''
def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([85, 0, 0])
    upperWhite = np.array([179, 160, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite

def nothing(a):
    pass

'''
Function to set track bar so that user can set the path points
    Arguments : 1. intial vales of trackbar
                2. Trackbar dimension 
'''

def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

'''
Function to get track bar values 
    Arguments : 1. Trackbar dimension 
'''

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points
'''
Function to get points from Grayscale image to track the lane 
    Arguments : 1. Input image
                2. Values form trackbar
                3. Widht and height of image 
                4. Inverse variable  
'''

def warpImg (img,points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp

'''
Function to marks the points of trackbar into image 
    Arguments : 1. input image 
                2. Values from trackbar
'''
def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

'''
Function to colloect and sum the values of array through the image to get lane 
    Arguments : 1. Input image
                2. Display order
                3. Varible Value intial to minimum percentage 
'''
def getHistogram(img, display=False, minPer=0.1, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else :
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)

    maxValue = np.max(histValues)  # FIND THE MAX VALUE
    minValue = minPer * maxValue
    indexArray = np.where(histValues >= minValue)  # ALL INDICES WITH MIN VALUE OR ABOVE
    basePoint = int(np.average(indexArray))  # AVERAGE ALL MAX INDICES VALUES

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-(intensity//255//region)),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist

    return basepoint

'''
Function to Display all windows tags together 
    Arguments : 1. Number od output display
                2. Array of image points
'''

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray&[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver