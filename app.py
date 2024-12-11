from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import warnings
import time
from flask_sock import Sock
import json
from queue import Queue
import threading
import base64
import re

app = Flask(__name__)
sock = Sock(app)

# Khởi tạo các biến và model giống như trong run.py
warnings.filterwarnings("ignore")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open('pose_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Columns đầy đủ
columns = ('x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4',
           'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7', 'z7', 'v7', 'x8', 'y8', 'z8', 'v8',
           'x9', 'y9', 'z9', 'v9', 'x10', 'y10', 'z10', 'v10', 'x11', 'y11', 'z11', 'v11', 'x12', 'y12', 'z12', 'v12',
           'x13', 'y13', 'z13', 'v13', 'x14', 'y14', 'z14', 'v14', 'x15', 'y15', 'z15', 'v15', 'x16', 'y16', 'z16',
           'v16', 'x17', 'y17', 'z17', 'v17', 'x18', 'y18', 'z18', 'v18', 'x19', 'y19', 'z19', 'v19', 'x20', 'y20',
           'z20', 'v20', 'x21', 'y21', 'z21', 'v21', 'x22', 'y22', 'z22', 'v22', 'x23', 'y23', 'z23', 'v23', 'x24',
           'y24', 'z24', 'v24', 'x25', 'y25', 'z25', 'v25', 'x26', 'y26', 'z26', 'v26', 'x27', 'y27', 'z27', 'v27',
           'x28', 'y28', 'z28', 'v28', 'x29', 'y29', 'z29', 'v29', 'x30', 'y30', 'z30', 'v30', 'x31', 'y31', 'z31',
           'v31', 'x32', 'y32', 'z32', 'v32', 'x33', 'y33', 'z33', 'v33')

fall_start_time = None
normal_start_time = None
fall_detected = False

# Thêm biến global để theo dõi trạng thái camera
camera_active = True

# Thêm biến global để lưu WebSocket connections
websocket_connections = set()
frame_queue = Queue(maxsize=10)
camera_thread = None
processing_thread = None

@sock.route('/ws')
def websocket(ws):
    websocket_connections.add(ws)
    try:
        while True:
            data = ws.receive()
    except:
        websocket_connections.remove(ws)

def notify_clients():
    print("Notifying clients about warning")
    message = json.dumps({"action": "play_warning"})
    dead_sockets = set()
    
    for ws in websocket_connections:
        try:
            ws.send(message)
            print(f"Warning sent successfully to client")
        except Exception as e:
            print(f"Error sending warning: {e}")
            dead_sockets.add(ws)
    
    websocket_connections.difference_update(dead_sockets)

def find_available_camera(max_index=10):
    cap = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            return cap
        else:
            cap.release()
    print("No available camera found.")
    return None

def camera_capture():
    global camera_active
    cap = find_available_camera()
    if not cap:
        return
    
    cap.set(3, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * 2)
    
    while camera_active:
        if not camera_active:
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        if not frame_queue.full():
            frame_queue.put(frame)
    
    cap.release()

def process_frame():
    global camera_active, fall_start_time, normal_start_time, fall_detected
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while camera_active:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Copy phần xử lý pose detection từ run.py
                body_pose = results.pose_landmarks.landmark
                pose_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_pose]).flatten())

                x = pd.DataFrame([pose_row], columns=columns)
                body_language_class = model.predict(x)[0]
                body_language_prob = model.predict_proba(x)[0]
                max_act = round(body_language_prob[np.argmax(body_language_prob)], 2)

                # Copy phần xử lý fall detection từ run.py
                global fall_start_time, normal_start_time, fall_detected
                if body_language_class == 'notgood' and max_act > 0.6:
                    if fall_start_time is None:
                        print("Fall detected, sending notification")
                        fall_start_time = time.time()
                        normal_start_time = None
                        notify_clients()
                    if time.time() - fall_start_time > 0.5:
                        fall_detected = True
                elif body_language_class == 'True':
                    if normal_start_time is None:
                        normal_start_time = time.time()
                        fall_start_time = None
                    if time.time() - normal_start_time > 3:
                        fall_detected = False

                # Vẽ detection box
                height, width = frame.shape[:2]
                landmark_points = np.array([(landmark.x * width, landmark.y * height) for landmark in body_pose])
                x, y, w, h = cv2.boundingRect(landmark_points.astype(int))
                color = (0, 0, 255) if fall_detected else (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                status = "Not Good" if fall_detected else "Good"
                cv2.putText(image, f"{status} ({max_act:.2f})", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    global camera_thread, processing_thread
    
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_capture)
        camera_thread.daemon = True
        camera_thread.start()
    
    return process_frame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera/<int:status>')
def toggle_camera(status):
    global camera_active
    camera_active = bool(status)
    
    if camera_active and (camera_thread is None or not camera_thread.is_alive()):
        camera_thread = threading.Thread(target=camera_capture)
        camera_thread.daemon = True
        camera_thread.start()
    
    return {'success': True}

@app.route('/process_frame', methods=['POST'])
def process_frame_from_browser():
    try:
        data = request.json
        # Xử lý base64 image data
        image_data = data['frame'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Chuyển đổi thành numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if not frame_queue.full():
            frame_queue.put(frame)
            
        return {'success': True}
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'success': False, 'error': str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
