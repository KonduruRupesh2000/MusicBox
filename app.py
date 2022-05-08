from flask import Flask,render_template,Response
import cv2
import numpy as np
from keras.models import model_from_json
from datetime import datetime
import time
import os,random
import subprocess

app=Flask(__name__)

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}
# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")
global text
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        # read the camera frame
        success,frame=camera.read()
        frame = cv2.flip(frame, 1, 0)
        if not success:
            break
        else:
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
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'+b'text')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('cam.html')

@app.route('/player')
def player():
    while True:
        try:  
        # read the camera frame
            success,frame=camera.read()
            frame = cv2.flip(frame, 1, 0)
            if not success:
                break
            else:
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
                if text=='Happy':
                    return render_template('happy.html')
                elif text=='Sad':
                    return render_template('sad.html')
                elif text=='Angry':
                    return render_template('angry.html')
                elif text=='Neutral':
                    return render_template('neutral.html')
                elif text=='Surprise':
                    return render_template('surprise.html')
        except:
            return render_template("404page.html")

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
