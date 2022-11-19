import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

#Look for the path
dataPath = 'C:/Users/jeres/Desktop/Proyecto_Final/image_data'
dataList = os.listdir(dataPath)
print('Lista de personas: ', dataList)

labels = []
data = []
label = 0

#Save every image in the path
for dataDir in dataList:
    data_Path = dataPath + '/' + dataDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(data_Path):
        print('Images: ', dataDir + '/' + fileName)
        labels.append(label)
        img_gray = cv2.imread(str(data_Path + '/' + fileName), 0)
        img_resize = cv2.resize(img_gray, (70, 70), interpolation=cv2.INTER_CUBIC)
        data.append(img_resize)
    label = label + 1

#Create numpy class and reshape
X = np.array(data)
X = X.reshape(X.shape + (1,))
y = np.reshape(np.array(labels), (-1, 1))

#print(y.shape)

#Split data in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

#Scale data
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

#print(X_train_scaled.shape)

#Create categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#print(y_train.shape)

#Get number of classes
n_classes = y_test.shape[1]

print(n_classes)

#Generate data
datagen = ImageDataGenerator(rotation_range=45,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.2, zoom_range=0.3,
                            horizontal_flip=True,
                            fill_mode="nearest")

#CNN model for training
def modelo_CNN():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(70, 70, 1), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Create model
model=modelo_CNN()

#Training
history = model.fit_generator(datagen.flow(X_train_scaled, y_train, batch_size=32),
	validation_data=(X_test_scaled, y_test), steps_per_epoch=len(X_train_scaled) // 32,
	epochs=16, verbose=1)

#Get scores of the model
scores=model.evaluate(X_test_scaled, y_test, verbose=0)

print('Error del modelo base : %.2f%%' % (100-scores[1]*100))

#Save model
model.save('C:/Users/jeres/Desktop/Proyecto_Final/Model_trained/new_full_model.h5')

#Look summary
model.summary()