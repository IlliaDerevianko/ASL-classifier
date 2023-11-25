import cv2
from keras import models
from keras.models import load_model
from tools import preprocess_image
import numpy as np
# import mediapipe as mp


model = load_model('MY_MODEL')
labels_list = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']
# mpHands = mp.solutions.hands
# mpDraw = mp.solutions.drawing_utils
# hands = mpHands.Hands()


capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    isTrue, frame = capture.read()
    # imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = hands.process(imgRGB)
    # print(results)
    # if results.multi_hand_landmarks:
    #     for handLms in results.multi_hand_landmarks:
    #         mpDraw.draw_landmarks(frame, handLms)

    cv2.rectangle(frame, (width // 2 - 200, height // 2 - 200), (width // 2 + 200, height // 2 + 200), (0, 255, 0), 2)
    roi = frame[height // 2 - 200:height // 2 + 200, width // 2 - 200:width // 2 + 200]
    # roi = cv2.imread('/Users/iladerevanko/Desktop/Python/ASL classifier/test_data/S_test.jpg')
    roi = preprocess_image(roi)
    roi = np.reshape(roi, (1, *roi.shape))
    probabilities = model.predict(roi)
    index = probabilities.argmax(axis=1)[0]
    confidence_level = probabilities[0][index] * 100
    prediction = labels_list[index] + ' ' + '{:.2f}'.format(confidence_level) + '%'
    cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('video', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv2.destroyAllWindows()