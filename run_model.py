import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from playsound import playsound
import threading

class DrowsinessDetector:
    def __init__(self, model_path='my_model.h5'):
        """
        Khởi tạo detector buồn ngủ
        
        Args:
            model_path: Đường dẫn đến file model .h5
        """
        print("🔄 Đang tải model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model đã sẵn sàng!")
        
        # Kích thước input của model
        self.img_size = (145, 145)
        
        # Load face detector từ OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load eye detector (để phát hiện mắt)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Ngưỡng cảnh báo
        self.drowsy_threshold = 0.5  # Nếu > 0.5 = buồn ngủ
        self.alert_frames = 0  # Đếm số frame liên tiếp buồn ngủ
        self.alert_limit = 15  # Cảnh báo sau 15 frame liên tiếp
        
        # Trạng thái
        self.is_playing_alert = False
        
    def preprocess_face(self, face_img):
        """
        Tiền xử lý ảnh khuôn mặt
        
        Args:
            face_img: Ảnh khuôn mặt từ webcam
            
        Returns:
            Ảnh đã xử lý sẵn sàng cho model
        """
        # Resize về kích thước model yêu cầu
        face_resized = cv2.resize(face_img, self.img_size)
        
        # Chuẩn hóa pixel về [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Thêm batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_drowsiness(self, face_img):
        """
        Dự đoán trạng thái buồn ngủ
        
        Args:
            face_img: Ảnh khuôn mặt
            
        Returns:
            prediction: Giá trị dự đoán (0-1)
            status: Trạng thái ("Tỉnh táo" hoặc "Buồn ngủ")
        """
        # Tiền xử lý
        processed_face = self.preprocess_face(face_img)
        
        # Dự đoán
        prediction = self.model.predict(processed_face, verbose=0)[0][0]
        
        # Xác định trạng thái
        if prediction > self.drowsy_threshold:
            status = "Buồn ngủ"
            color = (0, 0, 255)  # Đỏ
        else:
            status = "Tỉnh táo"
            color = (0, 255, 0)  # Xanh lá
            
        return prediction, status, color
    
    def play_alert_sound(self):
        """Phát âm thanh cảnh báo"""
        if not self.is_playing_alert:
            self.is_playing_alert = True
            # Tạo beep sound bằng OpenCV
            print("\n🚨 CẢNH BÁO: BUỒN NGỦ! 🚨")
            # Bạn có thể thêm file âm thanh:
            # threading.Thread(target=lambda: playsound('alert.mp3')).start()
            time.sleep(0.5)
            self.is_playing_alert = False
    
    def draw_info(self, frame, face_rect, prediction, status, color, fps):
        """
        Vẽ thông tin lên frame
        
        Args:
            frame: Frame video
            face_rect: Tọa độ khuôn mặt (x, y, w, h)
            prediction: Giá trị dự đoán
            status: Trạng thái
            color: Màu sắc
            fps: Frames per second
        """
        x, y, w, h = face_rect
        
        # Vẽ khung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Tạo background cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Hiển thị thông tin
        cv2.putText(frame, f"Trạng thái: {status}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Độ buồn ngủ: {prediction:.2%}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Thanh progress bar
        bar_width = int(prediction * 350)
        cv2.rectangle(frame, (20, 130), (370, 160), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 160), color, -1)
        
        # Cảnh báo nếu buồn ngủ
        if status == "Buồn ngủ":
            cv2.putText(frame, "⚠️ CẢNH BÁO! ⚠️", (x-20, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
    
    def run(self):
        """Chạy ứng dụng phát hiện buồn ngủ"""
        print("\n" + "="*60)
        print("🎥 PHÁT HIỆN BUỒN NGỦ QUA WEBCAM")
        print("="*60)
        print("📌 Hướng dẫn:")
        print("  - Nhìn thẳng vào camera")
        print("  - Nhấn 'q' để thoát")
        print("  - Nhấn 's' để chụp ảnh")
        print("="*60 + "\n")
        
        # Mở webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Không thể mở webcam!")
            return
        
        # Đặt độ phân giải
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS counter
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Không thể đọc frame từ webcam!")
                break
            
            # Lật frame để hiển thị như gương
            frame = cv2.flip(frame, 1)
            
            # Chuyển sang grayscale để detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện khuôn mặt
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            if len(faces) > 0:
                # Lấy khuôn mặt lớn nhất
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = face
                
                # Cắt khuôn mặt
                face_img = frame[y:y+h, x:x+w]
                
                # Dự đoán
                prediction, status, color = self.predict_drowsiness(face_img)
                
                # Đếm số frame buồn ngủ liên tiếp
                if status == "Buồn ngủ":
                    self.alert_frames += 1
                    if self.alert_frames >= self.alert_limit:
                        self.play_alert_sound()
                else:
                    self.alert_frames = 0
                
                # Vẽ thông tin
                frame = self.draw_info(frame, face, prediction, status, color, fps)
                
            else:
                # Không phát hiện được khuôn mặt
                cv2.putText(frame, "Không phát hiện khuôn mặt", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Tính FPS
            fps_counter += 1
            if fps_counter >= 10:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Hiển thị frame
            cv2.imshow('Phát hiện buồn ngủ - Nhấn Q để thoát', frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Đang thoát...")
                break
            elif key == ord('s'):
                # Chụp ảnh
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Đã lưu ảnh: {filename}")
        
        # Dọn dẹp
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Đã đóng webcam")


# ============================
# CHẠY CHƯƠNG TRÌNH
# ============================

if __name__ == "__main__":
    # Khởi tạo detector
    detector = DrowsinessDetector(model_path='my_model.h5')
    
    # Chạy
    detector.run()