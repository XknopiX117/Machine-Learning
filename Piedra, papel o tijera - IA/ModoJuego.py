import cv2
from more_itertools import first
import numpy as np
import tensorflow as tf

REV_CLASS_MAP = {
    'cgesture', 'like', 'paper', 'rock', 'scissors'
}

#Find the path of the trained model
filepath = 'C:/Users/jeres/Desktop/Proyecto_Final/Model_trained/new_full_model.h5'

#Load the model
model = tf.keras.models.load_model(filepath, compile=True)

#Start video capture
#0 for integrated webcam, 1 for external webcam
cap = cv2.VideoCapture(1)

#Starts the reado of the frames
ret, frame = cap.read()

#Flags
startFlag = None
backgroundFlag = False
firstFlag = True
winFlag = False
losesFlag = False
drawFlag = False

#Points
loses = 0
wins = 0
draw = 0

#Image that the computer will select
pc_img = 0

#Get the player's gesture
def mapSelection(player):

    if player[0, 2] == 1:
        return 2
    elif player[0, 3] == 1:
        return 3
    elif player[0, 4] == 1:
        return 4
    elif player[0, 0] == 1:
        return 0

#Look what the player and computer selected and who wins
def lookSelections(player, computer):
    global  loses, wins, draw
    global winFlag, drawFlag, losesFlag
    global backgroundFlag, startFlag
    #C: 0 rock, 1 paper, 2, scissors
    #P: 2 paper, 3 rock, 4 scissors
    if (computer == 0):
        if(player == 2):
            wins += 1
            winFlag = True
        elif(player == 3):
            draw += 1
            drawFlag = True
        elif(player == 4):
            loses += 1
            losesFlag = True
    # C: 0 rock, 1 paper, 2, scissors
    # P: 2 paper, 3 rock, 4 scissors
    elif (computer == 1):
        if (player == 2):
            draw += 1
            drawFlag = True
        elif (player == 3):
            loses += 1
            losesFlag = True
        elif(player == 4):
            wins += 1
            winFlag = True
    # C: 0 rock, 1 paper, 2, scissors
    # P: 2 paper, 3 rock, 4 scissors
    elif (computer == 2):
        if (player == 2):
            loses += 1
            losesFlag = True
        elif (player == 3):
            wins += 1
            winFlag = True
        elif(player == 4):
            draw += 1
            drawFlag = True

    if (player == 0):
        startFlag = False
        backgroundFlag = True
        winFlag = False
        losesFlag = False
        drawFlag = False
        wins = 0
        draw = 0
        loses = 0

#Load the image that the computer selected
def computerSelection(computer):
    global pc_img
    if computer == 0:
        pc_img = cv2.imread('C:/Users/jeres/Desktop/Proyecto_Final/computer_img/Rock.jpg')
    elif computer == 1:
        pc_img = cv2.imread('C:/Users/jeres/Desktop/Proyecto_Final/computer_img/paper.jpeg')
    else:
        pc_img = cv2.imread('C:/Users/jeres/Desktop/Proyecto_Final/computer_img/Scissors.jpeg')

#Counter for make a gesture
counter = 0

#Video capture loop
while cap.isOpened():

    #Read frame
    ret, frame = cap.read()

    #Draw a rectangle for the roi
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    #Ask for like gesture
    if startFlag == False and backgroundFlag == True:

        #Put some text
        cv2.putText(frame, 'Make like gesture to start', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Make C gesture to exit', (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        #Image preprocesing
        fmask = cv2.GaussianBlur(frame, (3, 3), 5)
        fmask = cv2.absdiff(background_img, fmask, 0)
        fmask = cv2.cvtColor(fmask, cv2.COLOR_BGR2GRAY)
        fmask = cv2.threshold(fmask, 25, 255, 0)[1]
        roi = fmask[100:400, 100:400]
        roi = cv2.resize(roi, (70, 70))
        roi = roi.reshape(70, 70, 1)

        #Get prediction
        prediction = model.predict(np.array([roi]))

        #Like gesture
        if prediction[0, 1] == 1:
            startFlag = True

        #C gesture
        if prediction[0, 0] == 1:
            cv2.destroyAllWindows()
            break

        print(prediction)

    #Like gesture has been detected, the game starts
    elif startFlag == True:

        #Put some text
        cv2.putText(frame, f'Time to choose: {100 - counter}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f'wins: {wins}, draw: {draw}, losses: {loses}', (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (127, 255, 127), 1,
                    cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        #Image preprocessing
        fmask = cv2.GaussianBlur(frame, (3, 3), 5)
        fmask = cv2.absdiff(background_img, fmask, 0)
        fmask = cv2.cvtColor(fmask, cv2.COLOR_BGR2GRAY)
        fmask = cv2.threshold(fmask, 25, 255, 0)[1]
        roi = fmask[100:400, 100:400]
        roi = cv2.resize(roi, (70, 70))
        roi = roi.reshape(70, 70, 1)

        #Get predict
        prediction = model.predict(np.array([roi]))
        print(f'Mi seleccion: {prediction}')

        #The counter finished
        if counter == 100:

            #Get prediction
            prediction = model.predict(np.array([roi]))
            print(f'Mi seleccion: {prediction}')
            counter = 0

            #Random computer select
            computerValue = np.random.randint(0, 2, 1)
            playerValue = mapSelection(prediction)

            #C gesture detected
            if playerValue == 0:
                cv2.destroyAllWindows()
                break

            #Other gesture detected, the game continues
            lookSelections(playerValue, computerValue)

            #Player won
            if winFlag == True:
                winFlag = False
                cv2.putText(frame, 'Winner', (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
            
            #Draw situation
            if drawFlag == True:
                drawFlag = False
                cv2.putText(frame, 'Draw', (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),
                            1, cv2.LINE_AA)

            #Computer won
            if losesFlag == True:
                losesFlag = False
                cv2.putText(frame, 'Lose', (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
                            1, cv2.LINE_AA)

            #Show what the computer selected
            cv2.imshow('Webcam', frame)
            computerSelection(computerValue)
            cv2.imshow('Computer selection', pc_img)
            print(f'Computer: {computerValue}')
            print(f'w: {wins}, d: {draw}, l: {loses}')
            cv2.waitKey(3000)
            cv2.destroyWindow('Computer selection')

        #Counter increse by 1, finish with 100
        counter += 1

    #Save a copy of the original frame for put some text and avoid
    #Overwrite in original
    if firstFlag == True:

        frame_copy = frame.copy()

        #Show frame and ask for key input
        cv2.putText(frame_copy, 'Press \'f\' key for inicial screenshot', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imshow('Starting game', frame_copy)
    k = cv2.waitKey(10)

    #If key 'f' was detected, it takes a screenshot of the background
    if k == ord('f'):

        #Copy the frame
        background_img = frame.copy()

        #Image preprocessing and quit the starting window
        background_img = cv2.GaussianBlur(background_img, (3, 3), 5)
        print("Background taken")
        cv2.destroyWindow('Starting game')
        firstFlag = False
        startFlag = False
        backgroundFlag = True

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()