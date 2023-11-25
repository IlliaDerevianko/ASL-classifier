import cv2
import mediapipe as mp
import math
from keras.models import load_model
from tools import preprocess_image, square_box, predict
import numpy as np


model = load_model('best_acc_model')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

gap = 35

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        left_corner = (math.inf, math.inf)
        right_corner = (-math.inf, -math.inf)

        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks[0].landmark:
                if landmark.x < left_corner[0]:
                    left_corner = (landmark.x, left_corner[1])
                if landmark.y < left_corner[1]:
                    left_corner = (left_corner[0], landmark.y)
                if landmark.x > right_corner[0]:
                    right_corner = (landmark.x, right_corner[1])
                if landmark.y > right_corner[1]:
                    right_corner = (right_corner[0], landmark.y)
            
            try:
                left_corner = (int(left_corner[0] * width) - gap, int(left_corner[1] * height) - gap)
                right_corner = (int(right_corner[0] * width) + gap, int(right_corner[1] * height) + gap)
                lc, rc = square_box(left_corner, right_corner)
                for corner in lc + rc:
                    if corner < 0:
                        raise Exception('Out of bounds!')
                if lc[0] > width or rc[0] > width or lc[1] > height or rc[1] > height:
                    raise Exception('Out of bounds!')
            except Exception:
                cv2.putText(image, 'Put your hand in the frame!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                lc, rc = square_box(left_corner, right_corner)
                cv2.rectangle(image, lc, rc, color=(0, 0, 255), thickness=2)
                roi = image[lc[1]:rc[1], lc[0]:rc[0]]
                prediction = predict(roi, model)
                cv2.putText(image, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #         drawing_styles.get_default_hand_landmark_style(),
                #         drawing_styles.get_default_hand_connection_style())

        cv2.imshow('Sign Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()