from flask import Flask, Response
import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db
import threading

app = Flask(__name__)

# ✅ Kết nối Camera IP
USERNAME = "khiem123"
PASSWORD = "khiem123"
IP_CAMERA = "192.168.1.18"
PORT = "554"
camera_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_CAMERA}:{PORT}/stream1"

# ✅ Kết nối Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://doan111111-default-rtdb.firebaseio.com'})

roi_ref = db.reference("roi")  # Tham chiếu đến Firebase

# ✅ Biến lưu vùng ROI (Mặc định)
roi_x, roi_y, roi_w, roi_h = 150, 300, 400, 100  

def update_roi(event):
    """Lắng nghe Firebase và cập nhật vùng ROI khi có thay đổi."""
    global roi_x, roi_y, roi_w, roi_h
    data = event.data  # Lấy dữ liệu từ Firebase
    if data:
        roi_x = data.get('x', roi_x)
        roi_y = data.get('y', roi_y)
        roi_w = data.get('w', roi_w)
        roi_h = data.get('h', roi_h)
        print(f"🔥 Vùng ROI cập nhật: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

# ✅ Chạy lắng nghe Firebase trong luồng riêng
threading.Thread(target=lambda: roi_ref.listen(update_roi), daemon=True).start()

def create_video_capture():
    """Tạo đối tượng VideoCapture với xử lý lỗi."""
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("❌ Không thể kết nối Camera IP! Kiểm tra lại.")
        return None
    return cap

def generate_frames():
    """Luồng phát video và phát hiện chuyển động."""
    global roi_x, roi_y, roi_w, roi_h  

    cap = create_video_capture()
    if cap is None:
        return

    while True:
        # 🔥 Cập nhật vùng ROI từ Firebase trong vòng lặp
        data = roi_ref.get()
        if data:
            roi_x = data.get('x', roi_x)
            roi_y = data.get('y', roi_y)
            roi_w = data.get('w', roi_w)
            roi_h = data.get('h', roi_h)

        success, frame2 = cap.read()
        if not success:
            print("⚠️ Mất kết nối, thử lại...")
            cap.release()
            time.sleep(5)
            cap = create_video_capture()
            if cap is None:
                continue

        # Xử lý ảnh xám
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # 🔥 **Sử dụng vùng ROI động từ Firebase**
        roi2 = gray2[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # So sánh với khung trước đó
        delta = cv2.absdiff(roi2, roi2)
        thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Tính % điểm trắng trong ROI
        white_pixels = np.sum(thresh == 255)
        total_pixels = roi_w * roi_h
        white_ratio = (white_pixels / total_pixels) * 100  

        # Đánh giá có chuyển động không
        motion_detected = white_ratio > 10  

        # Vẽ lại vùng ROI theo Firebase
        cv2.rectangle(frame2, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        status_text = f"QUAY ({white_ratio:.2f}%)" if motion_detected else f"DUNG YEN ({white_ratio:.2f}%)"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(frame2, status_text, (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', frame2)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    """API phát video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
