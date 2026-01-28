import cv2
import face_recognition
import numpy as np
import os
import dlib
import time
from scipy.spatial import distance


# EYE ASPECT RATIO FUNCTION

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# LOAD KNOWN FACES

known_face_encodings = []
known_face_names = []

path = "known_faces"

for file in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{file}")
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(file)[0])

print("[INFO] Known faces loaded")


# DLIB MODELS

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)


# LIVENESS PARAMETERS

EAR_THRESHOLD = 0.25
BLINK_FRAMES = 3
FACE_LIVENESS_TIME = 5  # seconds

blink_counter = 0
total_blinks = 0
face_start_time = None
is_real_face = False


# START CAMERA

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit()

print("[INFO] Camera started")


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(gray)


    # RESET WHEN NEW FACE APPEARS

    if len(faces) > 0 and face_start_time is None:
        face_start_time = time.time()
        total_blinks = 0
        blink_counter = 0
        is_real_face = False


    # BLINK DETECTION

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                total_blinks += 1
                is_real_face = True   # ✅ Only HERE face becomes real
            blink_counter = 0


    # LIVENESS DECISION

    if face_start_time is not None:
        elapsed_time = time.time() - face_start_time

        if is_real_face:
            liveness_label = "Real Face"
            color = (0, 255, 0)
        elif elapsed_time > FACE_LIVENESS_TIME:
            liveness_label = "Fake Face (Image)"
            color = (0, 0, 255)
        else:
            liveness_label = "Checking Liveness..."
            color = (0, 255, 255)
    else:
        liveness_label = "No Face"
        color = (255, 255, 255)

    cv2.putText(frame, liveness_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    # FACE RECOGNITION (ONLY IF REAL)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if is_real_face:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_face_names[best_match]

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    # RESET WHEN NO FACE

    if len(faces) == 0:
        face_start_time = None
        is_real_face = False

    cv2.imshow("Real-Time Face Recognition with Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
