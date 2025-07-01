# 💤 Sleep Detection System

This project uses **MediaPipe Face Mesh**, **OpenCV**, and **Pygame** to detect drowsiness based on **EAR (Eye Aspect Ratio)** and **MAR (Mouth Aspect Ratio)**. It features a simple **Tkinter GUI** for user interaction.

---

## ✅ Features

-   Eye closure detection using EAR
-   Yawning detection using MAR
-   Automatic audio alarm when drowsiness is detected
-   GUI with Start, Stop, and Exit buttons

---

## 🛠️ Installation

### 1. Clone or save the code

Save the main Python script as `sleep_detector.py` and place an audio file named `visual1.mp3` in the same directory.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

-   **Windows**:
    ```bash
    venv\Scripts\activate
    ```
-   **Linux/macOS**:
    ```bash
    source venv/bin/activate
    ```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy pygame
```

> Note: **Tkinter** is included by default in standard Python (CPython). If not, install it separately based on your OS.

-   **Linux**:

```bash
sudo apt-get install python-tk
**or**
pip3 install tk
```

---

## 📁 Project Structure

```
project_folder/
│
├── sleep_detector.py     # Main Python file
├── visual1.mp3           # Alarm sound file
└── README.md             # This file
```

---

## ▶️ How to Run

```bash
python sleep_detector.py
```

A GUI window will open with three buttons:

-   **Start** → Starts face and drowsiness detection
-   **Stop** → Temporarily stops detection
-   **Exit** → Closes the app and releases the webcam

---

## ⚙️ Important Parameters

| Parameter           | Value | Description                             |
| ------------------- | ----- | --------------------------------------- |
| `EAR_THRESH`        | 0.20  | Threshold to detect closed eyes         |
| `MAR_THRESH`        | 0.68  | Threshold to detect yawning             |
| `CLOSED_EYES_FRAME` | 17    | Number of frames indicating closed eyes |
| `YAWN_FRAME`        | 20    | Number of frames indicating a yawn      |

You can adjust these values in the code to fine-tune detection sensitivity.

---

## ❗ Notes

-   Ensure good lighting for accurate facial landmark detection.
-   Make sure your webcam is not in use by another application.
-   The alarm will stop automatically when the user opens their eyes or stops yawning.

---

## 👨‍💻 Author

Developed with Python 3.x and open-source libraries. Free to use and modify.

---
