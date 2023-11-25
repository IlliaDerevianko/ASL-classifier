from keras.layers.core import Flatten
from keras.saving.save import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from ASLNet import ASLNet
from tools import serialize_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


save_loc = '/Users/iladerevanko/Desktop/Python/ASL classifier/output'
dataset_loc = '/Users/iladerevanko/Desktop/Python/ASL classifier/ASL_alphabet'
weights_loc = 'MY_MODEL'
print('[INFO] loading dataset...')
trainX = np.load('serialized/X.npy')
trainY = np.load('serialized/y.npy')
testX = np.load('serialized/testX.npy')
testY = np.load('serialized/testY.npy')

labelNames = []
for loc in os.listdir(dataset_loc):
    if not loc.startswith('.'):
        labelNames.append(loc)
labelNames.sort()

print('[INFO] binarizing values...')
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print('[INFO] compiling model...')
opt = SGD(0.01, momentum=0.9)
model = ASLNet.build(100, 100, 3, classes=29)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint1 = ModelCheckpoint('best_loss_model', 'val_loss', mode="min", save_best_only=True, verbose=1)
checkpoint2 = ModelCheckpoint('best_acc_model', 'val_accuracy', mode='max', save_best_only=True, verbose=1)
callbacks = [checkpoint1, checkpoint2]

print('[INFO] training network...')
epochs = 30
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=epochs, verbose=1, callbacks=callbacks)
model.save(weights_loc)
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, epochs), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(save_loc)