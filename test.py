import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (based on MediaPipe Face Mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    coords = np.array([(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices])
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    A = euclidean_dist(coords[1], coords[5])
    B = euclidean_dist(coords[2], coords[4])
    C = euclidean_dist(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
EAR_THRESHOLD = 0.25
CLOSED_FRAMES_THRESHOLD = 30
counter = 0

cap = cv2.VideoCapture(0)
print("Starting webcam...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            counter += 1
            if counter > CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "DROWSY ALERT!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            counter = 0

        # Show EAR value
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
