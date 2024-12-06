from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

# Load the model
with open('test_again.pkl', 'rb') as file:
    model = pickle.load(file)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the fall detection variables
fall_start_time = None
normal_start_time = None
fall_detected = False

def find_available_camera(max_index=10):
    cap = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        else:
            cap.release()
    return None

cap = find_available_camera()

def generate_frames():
    global fall_detected, fall_start_time, normal_start_time
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                body_pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_pose]).flatten())
                x = pd.DataFrame([pose_row], columns=columns)
                body_language_class = model.predict(x)[0]
                body_language_prob = model.predict_proba(x)[0]
                max_act = round(body_language_prob[np.argmax(body_language_prob)], 2)

                if body_language_class == 'fall' and max_act > 0.4:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    elif time.time() - fall_start_time > 1:
                        fall_detected = True
                    normal_start_time = None
                else:
                    if fall_detected:
                        if normal_start_time is None:
                            normal_start_time = time.time()
                        elif time.time() - normal_start_time > 0.2:
                            fall_detected = False
                    fall_start_time = None

                # Draw detection box and get its coordinates
                draw_detection_box(image, body_pose, fall_detected)

            except Exception as e:
                print(e)
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
