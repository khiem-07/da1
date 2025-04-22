from flask import Flask, Response
import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db
import threading
import cv2.aruco as aruco  # Import ArUco

app = Flask(__name__)

# ✅ Kết nối Camera IP
USERNAME = "khiem123"
PASSWORD = "khiem123"
IP_CAMERA = "192.168.1.18"
PORT = "554"
camera_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_CAMERA}:{PORT}/stream1"

# ✅ Kết nối Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://doan111111-default-rtdb.firebaseio.com'
})
roi_ref = db.reference("roi")

# ✅ Biến vùng ROI mặc định (nếu không tìm được marker)
roi_x, roi_y, roi_w, roi_h = 150, 300, 400, 100

def update_roi(event):
    """Lắng nghe Firebase và cập nhật vùng ROI khi có thay đổi."""
    global roi_x, roi_y, roi_w, roi_h

    print(f"🔥 Dữ liệu từ Firebase: {event.data}")
    print(f"📍 Thay đổi tại đường dẫn: {event.path}")

    data = event.data
    path = event.path.lstrip("/")  # bỏ dấu "/" đầu

    if isinstance(data, dict):
        # Trường hợp ban đầu nhận full dict
        roi_x = data.get('x', roi_x)
        roi_y = data.get('y', roi_y)
        roi_w = data.get('w', roi_w)
        roi_h = data.get('h', roi_h)
        print(f"🔥 Vùng ROI cập nhật: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    elif isinstance(data, int) and path in ['x', 'y', 'w', 'h']:
        # Trường hợp chỉ cập nhật từng giá trị riêng lẻ
        if path == 'x':
            roi_x = data
        elif path == 'y':
            roi_y = data
        elif path == 'w':
            roi_w = data
        elif path == 'h':
            roi_h = data
        print(f"🔥 Vùng ROI cập nhật (mảnh): x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    else:
        print(f"⚠️ Dữ liệu Firebase không hợp lệ: {data}")


# ✅ Lắng nghe Firebase trong luồng riêng
threading.Thread(target=lambda: roi_ref.listen(update_roi), daemon=True).start()

def create_video_capture():
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("❌ Không thể mở camera IP!")
        return None
    return cap

def generate_frames():
    global roi_x, roi_y, roi_w, roi_h

    cap = create_video_capture()
    if cap is None:
        return

    prev_frame = None
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Mất kết nối, thử lại...")
            cap.release()
            time.sleep(5)
            cap = create_video_capture()
            if cap is None:
                continue
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 🔍 Phát hiện ArUco marker
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:
                    top_left = corners[i][0][0]
                    roi_x = int(top_left[0])
                    roi_y = int(top_left[1])
                    cv2.putText(frame, "ROI Updated (ArUco)", (roi_x, roi_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    break

        # ✅ Đảm bảo ROI không vượt giới hạn ảnh
        height, width = gray_blur.shape
        roi_x = min(max(0, roi_x), width - 1)
        roi_y = min(max(0, roi_y), height - 1)
        roi_w = min(roi_w, width - roi_x)
        roi_h = min(roi_h, height - roi_y)

        roi_gray = gray_blur[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # ✅ Bỏ qua nếu ROI trống
        if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
            print("⚠️ Vùng ROI không hợp lệ, bỏ qua frame.")
            continue

        # ✅ Khởi tạo hoặc kiểm tra kích thước khớp
        if prev_frame is None or prev_frame.shape != roi_gray.shape:
            prev_frame = roi_gray.copy()
            continue

        delta = cv2.absdiff(prev_frame, roi_gray)
        thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        white_pixels = np.sum(thresh == 255)
        total_pixels = roi_gray.shape[0] * roi_gray.shape[1]
        white_ratio = (white_pixels / total_pixels) * 100
        motion_detected = white_ratio > 10

        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        status_text = f"QUAY ({white_ratio:.2f}%)" if motion_detected else f"DUNG YEN ({white_ratio:.2f}%)"

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)
        cv2.putText(frame, status_text, (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        prev_frame = roi_gray.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
