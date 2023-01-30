# Read logo and resize
import cv2
import numpy as np

frame = cv2.imread('sword-removebg.png')
logo = cv2.imread('lightning/1.png')
size = 100
logo = cv2.resize(logo, (size, size))
  
# Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

roi = frame[-size-50:-50, -size-10:-10]
roi[np.where(mask)] = 0
roi += logo
cv2.imshow('WebCam', frame)
cv2.imwrite("Merge.png", frame)
cv2.waitKey(0)