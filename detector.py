import cv2
import time

import numpy
import pose_estimation_class as pm
import mediapipe as mp
import argparse



detector = pm.PoseDetector()

vid = cv2.VideoCapture(0)

sword = cv2.imread("sword-removebg.png")

ratio = sword.shape[1] / sword.shape[0]
sword = cv2.resize(sword, (int(100*ratio), int(100)))
sword = cv2.transpose(sword)
sword = numpy.flip(sword,axis = 0)
cv2.imshow("sword.jpg", sword)

while(True):
    ret, frame = vid.read()
    
    img, results = detector.findPose(frame, True)
    # img = img * 0
    
    # draw points
    mp.solutions.drawing_utils.draw_landmarks( 
        img,
        results.results.pose_landmarks, 
        results.mpPose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    HandPosition = detector.get_pointCoordinate(frame)
    if(HandPosition != -1): 
        LEFT = [int((HandPosition[0][0] + HandPosition[2][0]) / 2),int((HandPosition[0][1] + HandPosition[2][1]) / 2)]
        
        RIGHT = [int((HandPosition[1][0] + HandPosition[3][0]) / 2),int((HandPosition[1][1] + HandPosition[3][1]) / 2)]
        
        img = cv2.circle(img, (int (LEFT[0]),int(LEFT[1])), radius = 10, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (int (RIGHT[0]),int(RIGHT[1])), radius = 10, color=(0, 0, 255), thickness=-1)
        result_with_sword = img.copy()
        print(img.shape)
        print(sword.shape)
        print(RIGHT)
        print(min(img.shape[0],RIGHT[1] + sword.shape[0]) - RIGHT[1])
        print(min(img.shape[1],RIGHT[0] + sword.shape[1]) - RIGHT[0])
        print(sword[..., 0].min())
        print(sword[..., 0].max())
        RIGHT[0] = RIGHT[0] - int(sword.shape[0]/5)
        RIGHT[1] = RIGHT[1] - int(sword.shape[1] * 2)

        result_with_sword = img.copy()
        if(RIGHT[0] < img.shape[1] and RIGHT[1] < img.shape[0] and RIGHT[0] > 0 and RIGHT[1] > 0):
            result_with_sword[RIGHT[1]:min(img.shape[0],RIGHT[1] + sword.shape[0]),
                            RIGHT[0]:min(img.shape[1],RIGHT[0] + sword.shape[1])] = sword[
                            0:min(img.shape[0],RIGHT[1] + sword.shape[0]) - RIGHT[1],
                            0:min(img.shape[1],RIGHT[0] + sword.shape[1]) - RIGHT[0]]
            for i in range(RIGHT[1],min(img.shape[0],RIGHT[1] + sword.shape[0])):
                for j in range(RIGHT[0],min(img.shape[1],RIGHT[0] + sword.shape[1])):
                    for k in range(3):
                        if(result_with_sword[i,j,k] == 0):
                            result_with_sword[i,j,k] = img[i,j,k]
        cv2.imshow("Sword Hand Position", sword)
        cv2.imshow("Image POSE", img)
        cv2.imshow("Result", result_with_sword)
        cv2.waitKey(1)

