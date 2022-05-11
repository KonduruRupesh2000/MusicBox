import cv2
import numpy as np
from keras.models import model_from_json
from datetime import datetime
import time
import os,random
import subprocess

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation


emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")


global text
# start the webcam feed
segmentor = SelfiSegmentation()
img=cv2.imread('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/Blur_image.jpg')

cap = cv2.VideoCapture(0)

now = time.time()###For calculate seconds of video
future = now + 60


while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1, 0)
    frame = cv2.resize(frame, (460, 360))
    frame = segmentor.removeBG(frame, (114,114,114), threshold=0.8)
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        text=emotion_dict[maxindex]

    cv2.imshow('Emotion Detection', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


    key = cv2.waitKey(30)& 0xff
    if time.time() > future:##after 20second music will play
        try:
            cv2.destroyAllWindows()
            mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
            if text == 'Angry':
                randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Melody_songs/"))
                print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Melody_songs/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Happy':
                randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/party_songs/"))
                print('You are smiling :) ,I playing special song for you: ' + randomfile)
                file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/party_songs/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Fearful':
                randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Motivational_songs/"))
                print('You have fear of something ,I playing song for you: ' + randomfile)
                file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Motivational_songs/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Sad':
                randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Happy_songs/"))
                print('You are sad,dont worry:) ,I playing song for you: ' + randomfile)
                file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Happy_songs/' + randomfile)
                subprocess.call([mp, file])
            break
        except :
            print('Please stay focus in Camera frame atleast 15 seconds & run again this program:)')
            break




    if key==27:
        break
cap.release()
cv2.destroyAllWindows()

# import cv2
# import cvzone
# from cvzone.SelfiSegmentationModule import SelfiSegmentation

# segmentor = SelfiSegmentation()
# img_b = cv2.imread('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/Blur_image.jpg')
# capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break

#     frame = cv2.resize(frame,[512,512])
#     out = segmentor.removeBG(frame, img_b, threshold=0.8)

#     cv2.imshow('Frame', out)

#     keyboard = cv2.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break
# cv2.destroyAllWindows()
