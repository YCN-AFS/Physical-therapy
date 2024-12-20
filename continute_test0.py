#https://youtu.be/We1uB79Ci-w?si=ERSCMrFCC-37iAWo&t=2295
import csv
import cv2
import numpy as np
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


path = "dataset/train/normal"
first_time = True
class_name = "normal" #980

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2, static_image_mode=True) as pose:
    files = os.listdir(path)
    for i in files:
        frame = cv2.imread(path + "/" + i, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Make detection
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Export coordinates
        try:
            body_pose = results.pose_landmarks.landmark
            num_coords = len(body_pose)

            # mark = ['class']
            # for val in range(1, num_coords+1):
            #     mark += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]





            #extract pose landmarks
            pose_row = list(np.array([[landmarks.x, landmarks.y, landmarks.z, landmarks.visibility] for landmarks in body_pose]).flatten())

            pose_row.insert(0, class_name)


            # Create file csv

            with open('again.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(pose_row)

        except:
            pass

        #Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('FoxCode', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()