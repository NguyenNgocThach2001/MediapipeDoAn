import cv2
import mediapipe as mp
import time

class PoseDetector:

    def __init__(self, static_image_mode = False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=True,model_complexity=2,enable_segmentation=True,min_detection_confidence=0.5)
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img, self

    def get_pointCoordinate(self, img):
        image_hight, image_width, image_color = img.shape
        if(self.results.pose_landmarks.landmark is None):
            return -1
        x_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_PINKY].x * image_width
        y_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_PINKY].y * image_hight
        left_pinky = [x_coodinate,y_coodinate]
        x_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_PINKY].x * image_width
        y_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_PINKY].y * image_hight
        right_pinky = [x_coodinate,y_coodinate]
        x_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_INDEX].x * image_width
        y_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_INDEX].y * image_hight
        left_index = [x_coodinate,y_coodinate]
        x_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_INDEX].x * image_width
        y_coodinate = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_INDEX].y * image_hight
        right_index = [x_coodinate,y_coodinate]
        return [left_pinky, right_pinky, left_index, right_index]
    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList