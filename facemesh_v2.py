import cv2
import mediapipe as mp
import numpy as np
import time

# --------- Cài đặt FaceMesh ---------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark mắt (MediaPipe 468 points)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold
EAR_THRESH = 0.25
DROWSY_TIME = 2.0  # giây

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

# Thời gian nhắm mắt
eye_closed_start = None

# Hàm tính EAR
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in eye_indices])
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    ear = (A+B)/(2.0*C)
    return ear

# --------- Loop chính ---------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "No Face"

    if results.multi_face_landmarks:
        status = "Head Normal"
        lms = results.multi_face_landmarks[0].landmark

        # Vẽ landmark
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )

        # --------- EAR tính nhắm mắt ---------
        left_EAR = eye_aspect_ratio(lms, LEFT_EYE, w, h)
        right_EAR = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
        avg_EAR = (left_EAR + right_EAR)/2.0

        # --------- Head pose đơn giản ---------
        nose = np.array([lms[1].x*w, lms[1].y*h])
        left_eye = np.array([lms[33].x*w, lms[33].y*h])
        right_eye = np.array([lms[263].x*w, lms[263].y*h])
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        yaw = np.degrees(np.arctan2(dy, dx))
        pitch = (nose[1] - (left_eye[1]+right_eye[1])/2)  # đơn giản

        # --------- Phân loại trạng thái ---------
        if abs(yaw) > 20:
            status = "Head Turning"
        elif pitch > 15 and avg_EAR < EAR_THRESH:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > DROWSY_TIME:
                status = "Drowsiness"
        else:
            eye_closed_start = None

    # Hiển thị trạng thái
    cv2.putText(frame, f"Status: {status}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("FaceMesh Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
