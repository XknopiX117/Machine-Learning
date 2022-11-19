desc = '''Script to gather data images with a particular label.
Usage: python gather_images.py <label_name> <num_samples>
The script will collect <num_samples> number of images and store them
in its own directory.
Only the portion of the image within the box displayed
will be captured and stored.
Press 'a' to start/pause the image collecting process.
Press 'q' to quit.
'''

import cv2
import os
import sys

#Look for the carpet
try:
    label_name = 'rock'
    num_samples = 1500
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)

#Create path
IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

#Look for path
try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

#Start capture, 0 for integrated webacam, 1 for external
cap = cv2.VideoCapture(1)

start = False
count = 0

#Capture
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break

    #frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    if start:

        #Image preprocessing and roi
        frame = cv2.GaussianBlur(frame, (3, 3), 5)
        mask = cv2.absdiff(background, frame, 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 25, 255, 0)[1]

        roi = mask[100:400, 100:400]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    #Put collecting text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    #Look for selected key
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break
    if k == ord('f'):
        background = frame.copy()
        background = cv2.GaussianBlur(background, (3, 3), 5)
        print('Background taken, you can start the collecting')

#Close windows
print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
