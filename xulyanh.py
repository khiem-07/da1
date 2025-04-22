import cv2
import numpy as np

# Đường dẫn đến video của bạn
video_path = r"vid_xla.mp4"
output_path = r"out.mp4"  # Đường dẫn đầy đủ để lưu video đầu ra

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy thông tin video
fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy số khung hình trên giây
if fps == 0:  # Tránh lỗi nếu FPS không được xác định
    fps = 30  # Đặt mặc định thành 30 FPS
width = 360
height = 640

# Định nghĩa codec và tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Kích thước gấp đôi chiều rộng

# Danh sách tên cần in lên video
text_list = [
    "1. Bui Tran Anh Khoa - 21119219",
    
]

# Đọc video và xử lý từng khung hình
while True:
    ret, frame = cap.read()
    
    if not ret:  # Nếu không còn khung hình nào
        break

    # Thay đổi kích thước khung hình về 640x360
    frame_resized = cv2.resize(frame, (width, height))
    
    # Áp dụng bộ lọc Gaussian
    blurred_frame = cv2.GaussianBlur(frame_resized, (3, 3), 0)

    # Chuyển đổi sang hình ảnh xám
    gray_blurred = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    
    # Tạo mask nhị phân
    _, mask = cv2.threshold(gray_blurred, 200, 255, cv2.THRESH_BINARY)

    # Sử dụng erosion và dilation
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Tìm các đối tượng và gán nhãn
    num_labels, labels_im = cv2.connectedComponents(mask)

    # Vẽ bounding box cho từng đối tượng
    for label in range(1, num_labels):  # Bắt đầu từ 1 để bỏ qua nền
        y_indices, x_indices = np.where(labels_im == label)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

    # Chuyển đổi dilated_mask sang định dạng màu để ghép
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Ghép khung hình đã xử lý và khung hình gốc
    combined_frame = cv2.hconcat([frame_resized, mask_colored])  # Ghép theo chiều ngang

    # Vẽ danh sách tên lên video
    for i, text in enumerate(text_list):
        cv2.putText(combined_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Ghi khung hình đã xử lý vào video đầu ra
    out.write(combined_frame)

    # Hiển thị khung hình (có thể bỏ qua nếu không cần hiển thị)
    cv2.imshow('Video Processing', combined_frame)

    # Đợi phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video đã được lưu thành công:", output_path)