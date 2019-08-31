import cv2
import numpy as np
def nothing(x):
    pass

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.namedWindow("Trackbars")
cv2.createTrackbar("matrix","Trackbars",3,10,nothing)
#kernal= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
cv2.createTrackbar("lower","Trackbars",170,255,nothing)
cv2.createTrackbar("upper","Trackbars",255,255,nothing)

while(1):
    _,image = cam.read()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    low = cv2.getTrackbarPos("lower","Trackbars")
    up = cv2.getTrackbarPos("upper","Trackbars")
    _,thresh = cv2.threshold(gray,low,up,cv2.THRESH_BINARY)
    ############
    #erosion
    size = cv2.getTrackbarPos("matrix","Trackbars")
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    
    erosion = cv2.erode(thresh,kernal,iterations=1)
    #########
    #dilate
    dilate = cv2.dilate(thresh,kernal,iterations=1)
    

    _,contours,_ = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area>400:
            cv2.drawContours(image,[approx],0,(0),2)
            if len(approx)==3:
                cv2.putText(image,"TRIANGLE",(x,y),font,0.3,(0))
            elif len(approx)==4:
                cv2.putText(image,"RECTANGLE",(x,y),font,0.3,(0,255,0))
            elif len(approx)==5:
                cv2.putText(image,"PENTAGON",(x,y),font,0.3,(255,0,0))
            elif len(approx)>7 and len(approx)<9:
                cv2.putText(image,"CIRCLE",(x,y),font,0.3,(0))
            elif len(approx)>8:
                cv2.putText(image,"POLYGON",(x,y),font,0.3,(0))
    cv2.imshow("shape cam",image)
    cv2.imshow("threshold",thresh)
    #cv2.imshow("eroson",erosion)
    #cv2.imshow("dilate",dilate)

    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
