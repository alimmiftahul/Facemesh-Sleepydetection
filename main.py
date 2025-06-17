import cv2
import mediapipe as mp
import numpy as np
import pygame
import tkinter as tk

# Init pygame alarm
pygame.mixer.init()
pygame.mixer.music.load("visual1.mp3")

# Setup mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
EAR_THRESH = 0.20
MAR_THRESH = 0.68
CLOSED_EYES_FRAME = 17
YAWN_FRAME = 20
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# State
running = False
eye_closed_counter = 0
yawn_counter = 0

# Functions
def calculate_ear(landmarks, eye_idx, img_w, img_h):
    coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_idx]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, img_w, img_h):
    top = int(landmarks[13].y * img_h)
    bottom = int(landmarks[14].y * img_h)
    return abs(bottom - top) / img_h

def update_frame():
    global running, eye_closed_counter, yawn_counter

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if running:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks, w, h)

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if avg_ear < EAR_THRESH:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0

            if mar > MAR_THRESH:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if eye_closed_counter > CLOSED_EYES_FRAME or yawn_counter > YAWN_FRAME:
                cv2.putText(frame, "WAKE UP!", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Detection", frame)
    else:
        cv2.destroyWindow("Detection")

    if cv2.waitKey(1) & 0xFF == 27:
        exit_app()
        return

    root.after(10, update_frame)

def start_detection():
    global running
    running = True

def stop_detection():
    global running
    running = False
    pygame.mixer.music.stop()

def exit_app():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Init GUI
root = tk.Tk()
root.title("Sleep Detector")
root.geometry("600x400")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Sleep Detection System", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
title_label.pack(pady=20)

btn_font = ("Helvetica", 16)

btn_start = tk.Button(root, text="Start", width=35, height=3, font=btn_font,
                      bg="#ffa500", fg="black", command=start_detection)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop", width=35, height=3, font=btn_font,
                     bg="#333333", fg="black", command=stop_detection)
btn_stop.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", width=35, height=3, font=btn_font,
                     bg="#ff4d4d", fg="black", command=exit_app)
btn_exit.pack(pady=10)

# Start camera
cap = cv2.VideoCapture(0)

# Start update loop
root.after(10, update_frame)
root.mainloop()
