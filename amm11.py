import streamlit as st
import cv2
import numpy as np
import dlib
from simple_facerec import SimpleFacerec
import threading
import time
import webrtcvad
import pyaudio
from pynput import keyboard

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
sfr = SimpleFacerec()
sfr.load_encoding_images("face-recog/images")

# Initialize video capture
vs = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

# Initialize VAD and audio stream
vad = webrtcvad.Vad(1)
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)
stream.start_stream()

# Global variables
voice_detected = False
prohibited_keys = {
    frozenset([keyboard.Key.alt_l, keyboard.Key.tab]),
    frozenset([keyboard.Key.esc]),
    frozenset([keyboard.KeyCode.from_char('r')]),
    frozenset([keyboard.KeyCode.from_char('c')]),
}
pressed_keys = set()
key_violation = False

def calculate_focus(eye_aspect_ratio, head_angle, eye_closed, gaze_direction):
    EAR_THRESHOLD = 0.25
    HEAD_ANGLE_THRESHOLD = 5
    GAZE_THRESHOLD = 5
    IGNORE_GAZE_THRESHOLD = 10

    eye_focus = 1 - max(0, (eye_aspect_ratio - EAR_THRESHOLD) / EAR_THRESHOLD)
    if eye_closed:
        eye_focus = 0

    head_focus = max(0, 1 - (abs(head_angle) / HEAD_ANGLE_THRESHOLD))
    
    if abs(gaze_direction) > IGNORE_GAZE_THRESHOLD:
        gaze_focus = 1
    else:
        gaze_focus = max(0, 1 - abs(gaze_direction) / GAZE_THRESHOLD)

    focus_level = 0.4 * eye_focus + 0.3 * head_focus + 0.3 * gaze_focus
    return focus_level * 100

def get_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def get_gaze_direction(eye_points):
    eye_center = np.mean(eye_points, axis=0)
    eye_left = eye_points[0]
    eye_right = eye_points[3]
    direction = eye_center[0] - (eye_left[0] + eye_right[0]) / 2
    return direction

def process_voice():
    global voice_detected
    while True:
        try:
            audio_frame = stream.read(320, exception_on_overflow=False)
            voice_detected = vad.is_speech(audio_frame, 16000)
        except Exception as e:
            print(f"Error in voice detection: {e}")
        time.sleep(0.1)

def capture_frames():
    global vs, current_frame
    while True:
        ret, frame = vs.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(0.1)

def on_press(key):
    global key_violation
    pressed_keys.add(key)
    for combo in prohibited_keys:
        if combo.issubset(pressed_keys):
            key_violation = True
            print(f"Prohibited key combination detected: {', '.join(str(k) for k in combo)}")

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    st.title("Integrated Exam Monitoring System")

    option = st.sidebar.selectbox("Choose an option", ("Registration", "Exam"))

    if option == "Registration":
        st.header("User Registration")
        name = st.text_input("Enter your name")
        frame_placeholder = st.empty()
        register_button = st.button("Capture Photo")

        if name:
            threading.Thread(target=capture_frames, daemon=True).start()

            while True:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        continue

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_placeholder.image(buffer.tobytes(), channels="BGR")

                if register_button:
                    if name and current_frame is not None:
                        cv2.imwrite(f"face-recog/images/{name}.jpg", current_frame)
                        sfr.load_encoding_images("face-recog/images")
                        st.success(f"Registration successful for {name}!")
                        break
                    else:
                        st.error("Could not capture image. Please try again.")
                    time.sleep(1)

        st.write("Once registered, proceed to the exam section.")

    elif option == "Exam":
        st.header("Exam Monitoring")

        focus_placeholder = st.empty()
        frame_placeholder = st.empty()
        voice_indicator = st.empty()
        key_violation_indicator = st.empty()

        threading.Thread(target=capture_frames, daemon=True).start()
        threading.Thread(target=process_voice, daemon=True).start()
        threading.Thread(target=start_keyboard_listener, daemon=True).start()

        last_voice_update = time.time()
        while True:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            face_locations, face_names = sfr.detect_known_faces(frame)

            num_faces = len(face_locations)
            if num_faces > 1:
                st.warning("More than one person detected!")

            count_text = f"Persons Detected: {num_faces}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

            focus_level = 100  # Placeholder focus level
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                
                left_EAR = get_eye_aspect_ratio(left_eye)
                right_EAR = get_eye_aspect_ratio(right_eye)
                average_EAR = (left_EAR + right_EAR) / 2.0
                
                eye_closed = average_EAR < 0.25
                
                left_gaze = get_gaze_direction(left_eye)
                right_gaze = get_gaze_direction(right_eye)
                average_gaze = (left_gaze + right_gaze) / 2.0
                
                head_angle = 0  # Placeholder for head angle calculation
                
                focus_level = calculate_focus(average_EAR, head_angle, eye_closed, average_gaze)
                
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
                gaze_color = (0, 255, 0) if abs(average_gaze) < 10 else (0, 0, 255)
                cv2.line(frame, (face.left(), face.top()), (face.right(), face.bottom()), gaze_color, 2)
                
                focus_text = f"Focus: {focus_level:.2f}%"
                cv2.putText(frame, focus_text, (face.left(), face.bottom() + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            focus_placeholder.write(f"Overall Focus Level: {focus_level:.2f}%")

            current_time = time.time()
            if current_time - last_voice_update > 1:
                last_voice_update = current_time
                if voice_detected:
                    voice_indicator.markdown("**Voice Detected!**", unsafe_allow_html=True)
                else:
                    voice_indicator.markdown("No Voice Detected", unsafe_allow_html=True)

            if key_violation:
                key_violation_indicator.markdown("**Prohibited Key Press Detected!**", unsafe_allow_html=True)
            else:
                key_violation_indicator.markdown("No Key Violation Detected", unsafe_allow_html=True)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_placeholder.image(buffer.tobytes(), channels="BGR")
                
            time.sleep(0.1)

if __name__ == "__main__":
    main()
