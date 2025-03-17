import cv2
import numpy as np
import time
import subprocess  # Dùng để gọi chương trình C

# Đường dẫn video hoặc Camera IP
video_path = "rtsp://khiem123:khiem123@192.168.1.17:554/stream1"
  # Hoặc "rtsp://username:password@IP_CAMERA:PORT/stream"

# Mở video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()

# Đọc frame đầu tiên để lấy ROI
ret, frame1 = cap.read()
if not ret:
    print("❌ Lỗi khi lấy frame đầu tiên!")
    cap.release()
    exit()

# **THAY ĐỔI** các giá trị này để chọn vùng giám sát (ROI)
roi_x, roi_y, roi_w, roi_h = 150, 300, 400, 100  # (X, Y, Width, Height)

# Cắt vùng ROI từ frame1
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
roi1 = gray1[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# Biến theo dõi thời gian dừng
last_moving_time = time.time()
engine_stopped = False

while True:
    # Đọc frame tiếp theo
    ret, frame2 = cap.read()
    if not ret:
        print("✅ Video đã phát hết.")
        break

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
    motion_detected = white_ratio > 10  # Ngưỡng 3%

    # Xử lý thời gian dừng
    if motion_detected:
        engine_stopped = False  # Động cơ đang quay
        last_moving_time = time.time()
    else:
        if not engine_stopped:
            engine_stopped = True
            last_moving_time = time.time()
        elif time.time() - last_moving_time > 5:  # Nếu dừng quá 5 giây
            
            subprocess.run(["./warning"])  # Gọi chương trình C
            engine_stopped = False  # Reset để không gọi liên tục

    # Hiển thị khung hình và vùng ROI
    cv2.rectangle(frame2, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    status_text = f"QUAY ({white_ratio:.2f}%)" if motion_detected else f"DUNG YEN ({white_ratio:.2f}%)"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame2, status_text, (roi_x, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Giám sát khối trụ", frame2)
    cv2.imshow("Phát hiện chuyển động", thresh)

    # Nhấn 'q' để thoát
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Cập nhật frame trước
    roi1 = roi2.copy()

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
