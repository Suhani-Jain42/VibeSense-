import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

# Load trained model and label list
model = load_model("model.h5")
labels = np.load("labels.npy")

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

holis = holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("🎥 Real-time emotion recognition started... (Press ESC to exit)")

while True:
    ret, frm = cap.read()
    if not ret:
        print("❌ Could not access webcam.")
        break

    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = holis.process(rgb)

    landmarks = []

    if res.face_landmarks:
        base_x = res.face_landmarks.landmark[1].x
        base_y = res.face_landmarks.landmark[1].y

        # Face landmarks
        for lm in res.face_landmarks.landmark:
            landmarks.append(lm.x - base_x)
            landmarks.append(lm.y - base_y)

        # Left hand
        if res.left_hand_landmarks:
            base_x = res.left_hand_landmarks.landmark[8].x
            base_y = res.left_hand_landmarks.landmark[8].y
            for lm in res.left_hand_landmarks.landmark:
                landmarks.append(lm.x - base_x)
                landmarks.append(lm.y - base_y)
        else:
            landmarks.extend([0.0] * 42)  # 21 keypoints × 2

        # Right hand
        if res.right_hand_landmarks:
            base_x = res.right_hand_landmarks.landmark[8].x
            base_y = res.right_hand_landmarks.landmark[8].y
            for lm in res.right_hand_landmarks.landmark:
                landmarks.append(lm.x - base_x)
                landmarks.append(lm.y - base_y)
        else:
            landmarks.extend([0.0] * 42)

        # Ensure the input shape is correct
        if len(landmarks) == model.input_shape[1]:
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)
            predicted_label = labels[np.argmax(prediction)]

            # Display prediction
            cv2.putText(frm, f"Emotion: {predicted_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            cv2.putText(frm, "Incomplete landmarks", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Draw detected landmarks
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("Real-time Emotion Recognition", frm)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
