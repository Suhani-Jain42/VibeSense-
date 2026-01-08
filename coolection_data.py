import mediapipe as mp 
import numpy as np 
import cv2 
import pygetwindow as gw
import pyautogui
import time

# Function to bring OpenCV window to front
def bring_opencv_window_to_front():
    time.sleep(1)  # Give OpenCV time to show the window
    try:
        win = gw.getWindowsWithTitle("window")[0]
        win.activate()
        win.maximize()
    except Exception as e:
        print("Failed to bring OpenCV window to front:", e)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change to 1 if 0 doesn't work

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get user input
name = input("Enter the name of the data: ")

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

holis = holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Data storage
X = []
data_size = 0
window_brought_to_front = False  # Track whether we activated the window

while True:
    ret, frm = cap.read()
    if not ret:
        print("Failed to read from webcam")
        break

    frm = cv2.flip(frm, 1)

    # Process landmarks
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    lst = []

    if res.face_landmarks:
        base_x = res.face_landmarks.landmark[1].x
        base_y = res.face_landmarks.landmark[1].y

        for i in res.face_landmarks.landmark:
            lst.append(i.x - base_x)
            lst.append(i.y - base_y)

        if res.left_hand_landmarks:
            base_x = res.left_hand_landmarks.landmark[8].x
            base_y = res.left_hand_landmarks.landmark[8].y
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - base_x)
                lst.append(i.y - base_y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            base_x = res.right_hand_landmarks.landmark[8].x
            base_y = res.right_hand_landmarks.landmark[8].y
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - base_x)
                lst.append(i.y - base_y)
        else:
            lst.extend([0.0] * 42)

        X.append(lst)
        data_size += 1

    # Draw landmarks if available
    if res.face_landmarks:
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display frame
    cv2.putText(frm, f"Data size: {data_size}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("window", frm)

    # Bring window to front after first successful frame
    if data_size == 1 and not window_brought_to_front:
        bring_opencv_window_to_front()
        window_brought_to_front = True

    # Exit with ESC or after 100 samples
    if cv2.waitKey(1) == 27 or data_size > 99:
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()

# Save data
np.save(f"{name}.npy", np.array(X))
print(f"Data saved: {name}.npy with shape {np.array(X).shape}")

