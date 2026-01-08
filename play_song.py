import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os  # For program termination

# Load the trained model and label list
model = load_model("model.h5")
labels = np.load("labels.npy")

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
holis = holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Streamlit app UI
st.header("🎶 VibeSense- Detect Your Vibe")

# Session state to control camera
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Try loading last emotion from file
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Emotion detection from webcam
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = holis.process(rgb)

        features = []

        if res.face_landmarks:
            base_x = res.face_landmarks.landmark[1].x
            base_y = res.face_landmarks.landmark[1].y
            for lm in res.face_landmarks.landmark:
                features.append(lm.x - base_x)
                features.append(lm.y - base_y)

            if res.left_hand_landmarks:
                base_x = res.left_hand_landmarks.landmark[8].x
                base_y = res.left_hand_landmarks.landmark[8].y
                for lm in res.left_hand_landmarks.landmark:
                    features.append(lm.x - base_x)
                    features.append(lm.y - base_y)
            else:
                features.extend([0.0] * 42)

            if res.right_hand_landmarks:
                base_x = res.right_hand_landmarks.landmark[8].x
                base_y = res.right_hand_landmarks.landmark[8].y
                for lm in res.right_hand_landmarks.landmark:
                    features.append(lm.x - base_x)
                    features.append(lm.y - base_y)
            else:
                features.extend([0.0] * 42)

            if len(features) == model.input_shape[1]:
                input_data = np.array(features).reshape(1, -1)
                prediction = model.predict(input_data, verbose=0)
                pred_label = labels[np.argmax(prediction)]

                cv2.putText(frm, f"{pred_label}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

                np.save("emotion.npy", np.array([pred_label]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User input
lang = st.text_input("Enter language (e.g., English, Hindi):")
singer = st.text_input("Enter singer (optional):")

# Show webcam only when emotion not yet captured
if lang and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Recommend button
if st.button("🎵 Recommend Me Songs"):
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        query = f"{lang} {emotion} song {singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        np.save("emotion.npy", np.array([""]))  # Reset emotion
        st.session_state["run"] = "false"
        st.success("Opening songs on YouTube. Terminating the app...")
        st.stop()  # Stop further UI updates

        # Wait briefly to allow message to appear
        import time
        time.sleep(2)

        # Hard exit to stop the Streamlit server
        os._exit(0)
