import cv2
import time

import numpy
import pose_estimation_class as pm
import mediapipe as mp
import time

DRAW_GLOWING_LIGHT_TIME = 2
GLOWING_TIME = 2
TIME_TO_ADD_GLOWING_LIGHT_POINT = 0.01
TIME_TO_ADD_GLOWING_LIGHT_SUM = 0
GLOWING_LIGHT_TIME_COUNT = 0
TURN_ON_OFF_LIGHT = 1
ANIMATION_INDEX = 0
NUMBER_OF_ANIMATION = 1
detector = pm.PoseDetector()

vid = cv2.VideoCapture(0)

sword = cv2.imread("sword-removebg.png")

ratio = sword.shape[1] / sword.shape[0]
sword = cv2.resize(sword, (int(100*ratio), int(100)))
sword = cv2.transpose(sword)
sword = numpy.flip(sword,axis = 0)
cv2.imshow("sword.jpg", sword)

hand_light_queue = [[0,0,0,0]]

def add_glowing_light_point_to_queue(x, y, radius, time):
    hand_light_queue.append([x,y,radius, time])

start = time.time()
previous_frame = start
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
        #Sword
        RIGHT = [int((HandPosition[1][0] + HandPosition[3][0]) / 2),int((HandPosition[1][1] + HandPosition[3][1]) / 2)]
        img = cv2.circle(img, (int (RIGHT[0]),int(RIGHT[1])), radius = 10, color=(0, 0, 255), thickness=-1)
        result_with_sword = img.copy()
        # print(img.shape)
        # print(sword.shape)
        # print(RIGHT)
        # print(min(img.shape[0],RIGHT[1] + sword.shape[0]) - RIGHT[1])
        # print(min(img.shape[1],RIGHT[0] + sword.shape[1]) - RIGHT[0])
        # print(sword[..., 0].min())
        # print(sword[..., 0].max())
        rIGHT = RIGHT.copy()
        rIGHT[0] = rIGHT[0] - int(sword.shape[0]/7)
        rIGHT[1] = rIGHT[1] - int(sword.shape[1]*2)
        RIGHT[0] = RIGHT[0] - int(sword.shape[0]/5)
        RIGHT[1] = RIGHT[1] - int(sword.shape[1] * 2)
        result_with_sword = img.copy()
        # if(RIGHT[0] < img.shape[1] and RIGHT[1] < img.shape[0] and RIGHT[0] > 0 and RIGHT[1] > 0):
        #     result_with_sword[RIGHT[1]:min(img.shape[0],RIGHT[1] + sword.shape[0]),
        #                     RIGHT[0]:min(img.shape[1],RIGHT[0] + sword.shape[1])] = sword[
        #                     0:min(img.shape[0],RIGHT[1] + sword.shape[0]) - RIGHT[1],
        #                     0:min(img.shape[1],RIGHT[0] + sword.shape[1]) - RIGHT[0]]
        #     for i in range(RIGHT[1],min(img.shape[0],RIGHT[1] + sword.shape[0])):
        #         for j in range(RIGHT[0],min(img.shape[1],RIGHT[0] + sword.shape[1])):
        #             for k in range(3):
        #                 if(result_with_sword[i,j,k] == 0):
        #                     result_with_sword[i,j,k] = img[i,j,k]
        
        #NeonLight
        LEFT = [int((HandPosition[0][0] + HandPosition[2][0]) / 2),int((HandPosition[0][1] + HandPosition[2][1]) / 2)]
        img = cv2.circle(img, (int (LEFT[0]),int(LEFT[1])), radius = 10, color=(0, 0, 255), thickness=-1)

        now = time.time()
        delta_time = now - previous_frame
        previous_frame = now
        GLOWING_LIGHT_TIME_COUNT += delta_time
        TIME_TO_ADD_GLOWING_LIGHT_SUM += delta_time
        # print(delta_time)
        if(GLOWING_LIGHT_TIME_COUNT > DRAW_GLOWING_LIGHT_TIME):
            GLOWING_LIGHT_TIME_COUNT = 0
            ANIMATION_INDEX += 1
            ANIMATION_INDEX %= NUMBER_OF_ANIMATION

        if(TIME_TO_ADD_GLOWING_LIGHT_SUM > TIME_TO_ADD_GLOWING_LIGHT_POINT):
            TIME_TO_ADD_GLOWING_LIGHT_SUM = 0
            add_glowing_light_point_to_queue(rIGHT[0],rIGHT[1], 10, GLOWING_TIME)

        if(ANIMATION_INDEX == 0):
            for id in range(len(hand_light_queue)):
                result_with_sword = cv2.circle(result_with_sword, (hand_light_queue[id][0],hand_light_queue[id][1]), radius = hand_light_queue[id][2], color=(0, 0, 255), thickness=-1)
                hand_light_queue[id][1] =int(hand_light_queue[id][1] + delta_time * 20)
                hand_light_queue[id][2] =int(hand_light_queue[id][2] + delta_time * 10)
                hand_light_queue[id][3] -= delta_time
            hand_light_queue = [item for item in hand_light_queue if item[3] > 0]
        
        cv2.imshow("Sword Hand Position", sword)
        cv2.imshow("Image POSE", img)
        cv2.imshow("Result", result_with_sword)
        cv2.waitKey(1)

