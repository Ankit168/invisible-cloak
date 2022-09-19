from pickletools import uint8
import cv2 # importing OpenCV library of python
import numpy as np
import time

caputring_video = cv2.VideoCapture(0)
time.sleep(2) #allowwing camera to warmup
background = 0

# capturing the background in range of 60
# you should have video that have some seconds
# dedicated to background frame so that it 
# could easily save the background image
for i in range(60):
    return_val , background = caputring_video.read()
    
while(caputring_video.isOpened()):
    return_val , img = caputring_video.read()
    if not return_val:
        break
    
    #converting BGR to HSV for better 
    #detection or you can convert it to gray
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask_1 = cv2.inRange(hsv,lower_red,upper_red)
    
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv,lower_red,upper_red)
    
    mask_1 = mask_1 + mask_2
    
    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations =2)
    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations =1)
    
    mask_2 = cv2.bitwise_not(mask_1)
    
    result_1 = cv2.bitwise_and(background,background,mask=mask_1)
    result_2 = cv2.bitwise_and(img,img,mask=mask_2)
    final_output = cv2.addWeighted(result_1,1,result_2,1,0)
    cv2.imshow('Harry Potter Cloak',final_output)
    k = cv2.waitKey(10)
    if k==27:
        break
    
caputring_video.release()
cv2.destroyAllWindows()
    