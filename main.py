import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize alarm sound
pygame.mixer.init()
pygame.mixer.music.load("visual1.mp3")  # pastikan file ini ada di direktori

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR dan MAR threshold
# EAR_THRESH = 0.25
# MAR_THRESH = 0.7
# CLOSED_EYES_FRAME = 20
# YAWN_FRAME = 15
EAR_THRESH = 0.20 
MAR_THRESH = 0.68
CLOSED_EYES_FRAME = 17
YAWN_FRAME = 20

# Index ear and lips
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14]

# EAR function
def calculate_ear(landmarks, eye_idx, img_w, img_h):
    coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_idx]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# MAR function
def calculate_mar(landmarks, img_w, img_h):
    top = int(landmarks[13].y * img_h)
    bottom = int(landmarks[14].y * img_h)
    mar = abs(bottom - top) / img_h
    return mar

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
eye_closed_counter = 0
yawn_counter = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("❌ Tidak dapat membaca kamera.")
        break

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    image.flags.writeable = True
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Gambar FaceMesh
        for face_landmark in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # EAR dan MAR
        left_ear = calculate_ear(face_landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        mar = calculate_mar(face_landmarks, w, h)

        # Tampilkan nilai
        cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"MAR: {mar:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Logika deteksi
        if avg_ear < EAR_THRESH:
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0

        if mar > MAR_THRESH:
            yawn_counter += 1
        else:
            yawn_counter = 0

        if eye_closed_counter > CLOSED_EYES_FRAME or yawn_counter > YAWN_FRAME:
            cv2.putText(image, "WAKE UP!", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

    # Tampilkan output
    cv2.imshow("Drowsiness & Yawn Detection with FaceMesh", cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
