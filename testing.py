from re import I
from keras.models import load_model
import numpy as np
import cv2
import os
from tools import *
from imutils import paths
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import models
import matplotlib.pyplot as plt


model = load_model('MY_MODEL')

# img = cv2.imread('/Users/iladerevanko/Desktop/Python/ASL classifier/ASL_alphabet_1/A/5.jpg')
# print(img.dtype)
# img = preprocess_image(img)
# img = np.reshape(img, (1, *img.shape))
# cv2.imshow('img', img)
# cv2.waitKey(0)
# model_path = 'MY_MODEL'
# model = load_model(model_path)
# visualise_activations(img, model, 16)


# imagePaths = list(paths.list_images('test_set'))
# serialize_dataset(imagePaths, verbose=100)
# testX = np.load('serialized/testX.npy')
# testY = np.load('serialized/testY.npy')

# testX = testX.astype('float32') / 255.0
# lb = LabelBinarizer()
# testY = lb.fit_transform(testY)


# labels_list = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

# predictions = model.predict(testX)
# indexes = predictions.argmax(axis=1)
# out = [labels_list[i] for i in indexes]
# print(out)
# print(labels_list)
# real_labels = []

# test_data = []
# for loc in os.listdir('test_data'):
#     if not loc.startswith('.'):
#         test_data.append(os.path.join('test_data', loc))
#         real_labels.append(loc.split('.')[0])
# print(test_data)
# for i in range(len(test_data)):
#     img = cv2.imread(test_data[i])
#     img = cv2.resize(img, (100, 100))
#     img = img.astype('float32') / 255.0
#     test_data[i] = img
# test_data = np.array(test_data)

# dp = test_data[0]
# dp = np.reshape(dp, (1, 100, 100, 3))
# print(dp.shape)


# preds = model.predict(dp).argmax(axis=1)
# for pred in preds:
#     print(labels_list[pred])

