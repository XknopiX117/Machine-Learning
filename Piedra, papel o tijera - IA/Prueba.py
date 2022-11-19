import cv2
import numpy as np
import tensorflow as tf

REV_CLASS_MAP = {
    'cgesture', 'like', 'paper', 'rock', 'scissors'
}

filepath = 'C:/Users/jeres/Desktop/Proyecto_Final/Model_trained/new_full_model.h5'

model = tf.keras.models.load_model(filepath, compile=True)

cap = cv2.VideoCapture(1)
ret, frame = cap.read()

start = False
iniciar = False
counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)
    # Show image
    cv2.imshow('Webcam', frame)

    if counter == 15:

        counter = 0
        if iniciar == True:
            fmask = cv2.GaussianBlur(frame, (3, 3), 5)
            fmask = cv2.absdiff(fotoFondo, fmask, 0)
            fmask = cv2.cvtColor(fmask, cv2.COLOR_BGR2GRAY)
            fmask = cv2.threshold(fmask, 25, 255, 0)[1]
            #cv2.imshow('threshold', fmask)
            roi = fmask[100:400, 100:400]
            roi = cv2.resize(roi, (70, 70))
            roi = roi.reshape(70, 70, 1)
            cv2.imwrite("Foto.jpg", roi)

            start = False
            prediction = model.predict(np.array([roi]))
            print(prediction)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    elif k == ord('q'):
        fotoFondo = frame.copy()
        fotoFondo = cv2.GaussianBlur(fotoFondo, (3, 3), 5)
        print("Fondo tomado")
        iniciar = True

    # Checks whether q has been hit and stops the loop
    if k % 256 == 27:
        break
    counter += 1

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()
