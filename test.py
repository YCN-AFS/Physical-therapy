import cv2
import mediapipe as mp
# import pickle
# import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

path = '04-tuthedung.jpg'

# with open('test_again.pkl', 'rb') as file:
#     model = pickle.load(file)

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

def draw_detection_box(image, landmarks, is_fall):
    landmark_points = np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
    x, y, w, h = cv2.boundingRect(landmark_points.astype(int))
    color = (0, 0, 255) if is_fall else (0, 255, 0)  # Đỏ nếu té ngã, xanh lá nếu bình thường
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


    status = "Fall detected!" if is_fall else "Good"
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y - 10 if y - 10 > text_size[1] else y + text_size[1]
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return x, y, w, h

fall_start_time = None
normal_start_time = None
fall_detected = False



# Đọc ảnh từ file (thay đổi đường dẫn tới ảnh của bạn)
image_path = 'image/'+ path  # Thay đổi đường dẫn tới ảnh của bạn
image = cv2.imread(image_path)

height, width = image.shape[:2]

# Thay đổi kích thước ảnh xuống một nửa
new_width = int(width * 0.5)
new_height = int(height * 0.5)
image = cv2.resize(image, (new_width, new_height))

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    print("Không thể đọc ảnh.")
else:
    height, width = image.shape[:2]

    # Xử lý ảnh
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Initialize the pose detection
    pose = mp_pose.Pose()

    # Thực hiện phát hiện
    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Xuất tọa độ
    try:
        body_pose = results.pose_landmarks.landmark
        pose_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_pose]).flatten())

        # Vẽ hộp phát hiện và lấy tọa độ của nó
        box_x, box_y, box_w, box_h = draw_detection_box(image, body_pose, fall_detected)

    except Exception as e:
        print(e)
        pass

    # Vẽ các điểm khung xương
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hiển thị ảnh
    cv2.imshow('Fall Detection', image)
    # cv2.waitKey(0)  # Chờ cho đến khi nhấn phím để đóng cửa sổ
    cv2.imwrite('news/'+ path, image)
