import csv
import cv2
import numpy as np
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Directory containing subdirectories of images
IMAGE_DIR = "train"
FILE_NAME = "data.csv"

# Get class names from folder names
class_names = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]


# Function to process and save pose landmarks
def process_images():
    first_time = True

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        for class_name in class_names:
            class_dir = os.path.join(IMAGE_DIR, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                results = pose.process(image_rgb)

                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                try:
                    body_pose = results.pose_landmarks.landmark
                    num_coords = len(body_pose)

                    # Prepare CSV header
                    if first_time:
                        mark = ['class']
                        for val in range(1, num_coords + 1):
                            mark += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
                        with open(FILE_NAME, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(mark)
                        first_time = False

                    # Prepare pose row
                    pose_row = list(np.array(
                        [[landmarks.x, landmarks.y, landmarks.z, landmarks.visibility] for landmarks in
                         body_pose]).flatten())
                    pose_row.insert(0, class_name)

                    # Write pose row to CSV
                    with open(FILE_NAME, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(pose_row)

                except:
                    pass

                mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('Pose Estimation', image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()


# Run the image processing function
process_images()
