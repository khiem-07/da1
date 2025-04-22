import cv2
import numpy as np

# Cấu hình Camera IP (THAY THÔNG TIN ĐÚNG CỦA BẠN!)
USERNAME = "khiem123"
PASSWORD = "khiem123"
IP_CAMERA = "192.168.1.19"
PORT = "554"

# URL RTSP (CẬP NHẬT THEO CAMERA CỦA BẠN)
camera_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_CAMERA}:{PORT}/stream1"

# Kết nối Camera IP
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("❌ Không thể kết nối với Camera IP! Kiểm tra lại địa chỉ IP và thông tin đăng nhập.")
    exit()

# Đọc frame đầu tiên
ret, frame1 = cap.read()
if not ret:
    print("❌ Lỗi khi lấy frame đầu tiên từ camera!")
    cap.release()
    exit()




while True:
    # Đọc frame tiếp theo
    ret, frame2 = cap.read()
    if not ret:
        print("❌ Lỗi khi đọc frame từ Camera IP!")
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Tính toán sự khác biệt giữa frame hiện tại và frame trước đó
    delta = cv2.absdiff(frame1, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Tìm các vùng chuyển động
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Bỏ qua chuyển động nhỏ
            continue

        # Vẽ hình chữ nhật quanh vùng chuyển động
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị video trên laptop
    cv2.imshow("Camera IP - Phát hiện chuyển động", frame2)
    cv2.imshow("Phát hiện chuyển động", thresh)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Cập nhật frame trước
    frame1 = gray.copy()

# Giải phóng camera
cap.release()


cv2.destroyAllWindows()
