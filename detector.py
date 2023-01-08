import cv2
import time

import numpy
import pose_estimation_class as pm
import random
import mediapipe as mp
import time

DRAW_GLOWING_LIGHT_TIME = 3
GLOWING_TIME = 1
TIME_TO_ADD_GLOWING_LIGHT_POINT = 0.01
TIME_TO_ADD_GLOWING_LIGHT_SUM = 0
GLOWING_LIGHT_TIME_COUNT = 0
TURN_ON_OFF_LIGHT = 1
ANIMATION_INDEX = 0
NUMBER_OF_ANIMATION = 4
detector = pm.PoseDetector()

vid = cv2.VideoCapture(0)

sword = cv2.imread("sword-removebg.png")

ratio = sword.shape[1] / sword.shape[0]
sword = cv2.resize(sword, (int(100*ratio), int(100)))
sword = cv2.transpose(sword)
sword = numpy.flip(sword,axis = 0)
cv2.imshow("sword.jpg", sword)

hand_light_queue = []

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
            if(ANIMATION_INDEX == 0 or ANIMATION_INDEX == 2):
                add_glowing_light_point_to_queue(rIGHT[0],rIGHT[1], 10, GLOWING_TIME)
        result = img.copy()
        if(ANIMATION_INDEX == 0):
            for id in range(len(hand_light_queue)):
                if(id == 0):
                    hand_light_queue[id][3] -= delta_time * 2
                    continue
                num1 = int(random.randrange(255))
                num2 = int(random.randrange(255))
                num3 = int(random.randrange(255))
                print(num1,num2,num3)
                result_with_sword = cv2.line(result_with_sword, (hand_light_queue[id - 1][0],hand_light_queue[id - 1][1]), (hand_light_queue[id][0],hand_light_queue[id][1]), color=(num1, num2, num3), thickness=hand_light_queue[id][2])
                hand_light_queue[id][1] =int(hand_light_queue[id][1] + delta_time * 20)
                hand_light_queue[id][2] = max(1,hand_light_queue[id][2] - int(hand_light_queue[id][2] * 0.2))
                hand_light_queue[id][3] -= delta_time * 2
            hand_light_queue = [item for item in hand_light_queue if item[3] > 0]
            result = result_with_sword
        
        if(ANIMATION_INDEX == 1 or ANIMATION_INDEX == 3):
            context = str(int(GLOWING_LIGHT_TIME_COUNT) + 1)
            num1 = int(random.randrange(255))
            num2 = int(random.randrange(255))
            num3 = int(random.randrange(255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (100, 100)
            fontScale = 2
            color = (num1,num2,num3)
            thickness = 5
            result = cv2.putText(result_with_sword, context, org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)

        if(ANIMATION_INDEX == 2):
            for id in range(len(hand_light_queue)):
                num1 = int(random.randrange(255))
                num2 = int(random.randrange(255))
                num3 = int(random.randrange(255))
                result_with_sword = cv2.circle(result_with_sword, (int(hand_light_queue[id][0] + num1 / 50 - num2  / 50), int(hand_light_queue[id][1] + num2  / 50 - num3  / 50)), color=(num1, num2, num3), thickness=-1, radius = int(10))
                hand_light_queue[id][2] =max(1,hand_light_queue[id][2] - int(hand_light_queue[id][2] * 0.2))
                hand_light_queue[id][3] -=delta_time * 2
            hand_light_queue = [item for item in hand_light_queue if item[3] > 0]
            result = result_with_sword
        


        cv2.imshow("Sword Hand Position", sword)
        cv2.imshow("Image POSE", img)
        cv2.imshow("Result", result)
        cv2.waitKey(1)

