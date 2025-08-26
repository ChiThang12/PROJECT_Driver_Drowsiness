import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

# -----------------------------
# Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# EAR function
# -----------------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------------
# Head pose function
# -----------------------------
def get_head_pose(landmarks, frame_shape):
    # 2D image points từ Face Mesh
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[33],   # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[78],   # Left mouth corner
        landmarks[308]   # Right mouth corner
    ], dtype=np.float64)

    # 3D model points (approximate)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0),# Left eye
        (225.0, 170.0, -135.0), # Right eye
        (-150.0, -150.0, -125.0),# Left mouth
        (150.0, -150.0, -125.0) # Right mouth
    ], dtype=np.float64)

    size = frame_shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [float(degrees(angle)) for angle in euler_angles]
    return pitch, yaw, roll

# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3
blink_counter = 0
sleepy = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    frame_h, frame_w = frame.shape[:2]

    if results.multi_face_landmarks:
        sleepy = False
        for face_landmarks in results.multi_face_landmarks:
            # Chuyển landmark sang array 2D
            landmark_points = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                landmark_points.append((x,y))
            landmark_points = np.array(landmark_points)

            # EAR
            left_eye = landmark_points[[33,160,158,133,153,144]]
            right_eye = landmark_points[[263,387,385,362,380,373]]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter >= EAR_CONSEC_FRAMES:
                    sleepy = True
            else:
                blink_counter = 0

            # Head pose
            try:
                pitch, yaw, roll = get_head_pose(landmark_points, frame.shape)
                cv2.putText(frame, f"Pitch:{pitch:.1f} Yaw:{yaw:.1f} Roll:{roll:.1f}", 
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            except:
                pass

            # Vẽ Face Mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

    else:
        cv2.putText(frame, "No Face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
        blink_counter = 0
        sleepy = False

    # Hiển thị trạng thái
    if sleepy:
        cv2.putText(frame, "Sleepy", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)

    cv2.imshow("Face Mesh + EAR + Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
