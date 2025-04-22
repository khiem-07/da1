from flask import Flask, Response
import cv2
import numpy as np
import time
import subprocess  # Dùng để gọi chương trình C khi động cơ dừng

app = Flask(__name__)

# Cấu hình Camera IP
USERNAME = "khiem123"
PASSWORD = "khiem123"
IP_CAMERA = "192.168.1.17"
PORT = "554"

# URL RTSP của Camera
camera_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_CAMERA}:{PORT}/stream1"

def create_video_capture():
    """Tạo đối tượng VideoCapture với xử lý lỗi."""
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("❌ Không thể kết nối với Camera IP! Kiểm tra lại.")
        return None
    return cap

def generate_frames():
    """Luồng phát video và phát hiện chuyển động."""
    cap = create_video_capture()
    if cap is None:
        return

    # Lấy frame đầu tiên để so sánh
    ret, frame1 = cap.read()
    if not ret:
        print("❌ Lỗi khi lấy frame đầu tiên!")
        cap.release()
        return
    
    # **THAY ĐỔI** thông số này để chọn vùng giám sát (ROI)
    roi_x, roi_y, roi_w, roi_h = 150, 300, 400, 100  # (X, Y, Width, Height)

    # Cắt vùng ROI từ frame1
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
    roi1 = gray1[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Biến theo dõi trạng thái động cơ
    last_moving_time = time.time()
    engine_stopped = False

    while True:
        if cap is None or not cap.isOpened():
            print("⚠️ Mất kết nối, thử lại...")
            cap = create_video_capture()
            if cap is None:
                time.sleep(5)
                continue

        success, frame2 = cap.read()
        if not success:
            print("⚠️ Lỗi khi đọc frame, thử lại...")
            cap.release()
            cap = None
            time.sleep(5)
            continue

        # Xử lý ảnh xám và làm mịn
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Cắt vùng ROI từ frame hiện tại
        roi2 = gray2[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # So sánh sự thay đổi giữa 2 frame
        delta = cv2.absdiff(roi1, roi2)
        thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Tính % đốm trắng trong ROI
        white_pixels = np.sum(thresh == 255)
        total_pixels = roi_w * roi_h
        white_ratio = (white_pixels / total_pixels) * 100  # Tính theo %

        # Kiểm tra xem động cơ có quay không
        motion_detected = white_ratio > 10  # Ngưỡng 10%

        # Xử lý thời gian dừng
        if motion_detected:
            engine_stopped = False  # Động cơ đang quay
            last_moving_time = time.time()
        else:
            if not engine_stopped:
                engine_stopped = True
                last_moving_time = time.time()
            elif time.time() - last_moving_time > 5:  # Nếu dừng quá 5 giây
                print("⚠️ ĐỘNG CƠ DỪNG QUÁ LÂU! GỌI CẢNH BÁO ⚠️")
                subprocess.run(["./warning"])  # Gọi chương trình C
                engine_stopped = False  # Reset tránh gọi liên tục

        # Vẽ khung vùng ROI
        cv2.rectangle(frame2, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        status_text = f"QUAY ({white_ratio:.2f}%)" if motion_detected else f"DUNG YEN ({white_ratio:.2f}%)"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(frame2, status_text, (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Nén frame thành JPEG để truyền qua Flask
        _, buffer = cv2.imencode('.jpg', frame2)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Cập nhật frame trước
        roi1 = roi2.copy()

@app.route('/video')
def video_feed():
    """Định tuyến Flask để truyền video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
