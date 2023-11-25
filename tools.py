import os
import cv2
from keras.saving.save import load_model
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from keras import models
from imutils import paths
import sys



def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharp = cv2.filter2D(grayImage, -1, kernel)
    img = img.astype('float32') / 255
    return img

def square_box(left_corner, right_corner):
    roi_width = right_corner[0] - left_corner[0]
    roi_height = right_corner[1] - left_corner[1]
    # print('width: ', roi_width)
    # print('height: ', roi_height)
    if roi_width > roi_height:
        margin = (roi_width - roi_height) // 2
        left_corner = (left_corner[0], left_corner[1] - margin)
        right_corner = (right_corner[0], left_corner[1] + roi_width)
    elif roi_height > roi_width:
        margin = (roi_height - roi_width) // 2
        left_corner = (left_corner[0] - margin, left_corner[1])
        right_corner = (left_corner[0] + roi_height, right_corner[1])
    return left_corner, right_corner


def serialize_dataset(data_locations, verbose=1):
    imagePaths = []
    for loc in data_locations:
        imagePaths += list(paths.list_images(loc))
    shuffle(imagePaths)


    m = len(imagePaths)
    data = np.ones((m, 100, 100, 3), dtype='float32')
    labels = []
    
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        image = preprocess_image(image)
        if imagePath.split(os.path.sep)[-3] == 'ASL_alphabet_1' and imagePath.split(os.path.sep)[-2] != 'Z':
            image = cv2.flip(image, 1)

        label = imagePath.split(os.path.sep)[-2]
        data[i] = image
        labels.append(label)

        if verbose > 0 and i > 0 and ((i + 1) % verbose == 0 or i == m - 1):
            print(f'[INFO] processed {i + 1}/{m}')
    
    print('[INFO] serializing to the disk...')
    np.save('serialized/X', data)
    np.save('serialized/y', np.array(labels))
    print('[INFO] dataset was serialized successfully')

def visualise_activations(img, model, layers_num):
    layer_outputs = [layer.output for layer in model.layers[:layers_num]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    layer_names = []
    for layer in model.layers[:layers_num]:
        layer_names.append(layer.name)
    images_per_row = 16
    e = 10e-7
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= (channel_image.std() + e)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

def predict(roi, model):
    labels_list = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']
    roi = preprocess_image(roi)
    roi = np.reshape(roi, (1, *roi.shape))
    probabilities = model.predict(roi)
    index = probabilities.argmax(axis=1)[0]
    confidence_level = probabilities[0][index] * 100
    prediction = labels_list[index] + ' ' + '{:.2f}'.format(confidence_level) + '%'
    return prediction

if __name__ == '__main__':
    dataset_locations = ['ASL_alphabet_1']
    # dataset_locations = ['test_set', 'test_set_1']
    serialize_dataset(dataset_locations, verbose=5000)
