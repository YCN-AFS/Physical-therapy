import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open('test_again.pkl', 'rb') as file:
    model = pickle.load(file)

columns = ('x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4',
           'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7', 'z7', 'v7', 'x8', 'y8', 'z8', 'v8',
           'x9', 'y9', 'z9', 'v9', 'x10', 'y10', 'z10', 'v10', 'x11', 'y11', 'z11', 'v11', 'x12', 'y12', 'z12', 'v12',
           'x13', 'y13', 'z13', 'v13', 'x14', 'y14', 'z14', 'v14', 'x15', 'y15', 'z15', 'v15', 'x16', 'y16', 'z16',
           'v16',
           'x17', 'y17', 'z17', 'v17', 'x18', 'y18', 'z18', 'v18', 'x19', 'y19', 'z19', 'v19', 'x20', 'y20', 'z20',
           'v20',
           'x21', 'y21', 'z21', 'v21', 'x22', 'y22', 'z22', 'v22', 'x23', 'y23', 'z23', 'v23', 'x24', 'y24', 'z24',
           'v24',
           'x25', 'y25', 'z25', 'v25', 'x26', 'y26', 'z26', 'v26', 'x27', 'y27', 'z27', 'v27', 'x28', 'y28', 'z28',
           'v28',
           'x29', 'y29', 'z29', 'v29', 'x30', 'y30', 'z30', 'v30', 'x31', 'y31', 'z31', 'v31', 'x32', 'y32', 'z32',
           'v32',
           'x33', 'y33', 'z33', 'v33')


# Initialize the fall detection variables


fall_start_time = None
normal_start_time = None
fall_detected = False

# cap = cv2.VideoCapture(0)

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

cap = find_available_camera()

cap.set(3, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * 2)
ret, frame = cap.read()
height, width = frame.shape[:2]


def draw_detection_box(image, landmarks, is_fall):
    landmark_points = np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
    x, y, w, h = cv2.boundingRect(landmark_points.astype(int))
    color = (0, 0, 255) if is_fall else (0, 255, 0)  # Đỏ nếu té ngã, xanh lá nếu bình thường
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


    status = "Fall detected!" if is_fall else "Normal"
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y - 10 if y - 10 > text_size[1] else y + text_size[1]
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return x, y, w, h


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Export coordinates
        try:
            body_pose = results.pose_landmarks.landmark
            pose_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_pose]).flatten())

            # Make predictions
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
            box_x, box_y, box_w, box_h = draw_detection_box(image, body_pose, fall_detected)

            # Display confidence
            cv2.putText(image, f"Confidence: {max_act}", (box_x, box_y + box_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Fall Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()