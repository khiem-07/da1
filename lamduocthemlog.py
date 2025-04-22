import cv2
import numpy as np
from motpy import Detection, MultiObjectTracker
import logging
import datetime
import os
# Tạo thư mục logs nếu chưa tồn tại
if not os.path.exists('logs'):
    os.makedirs('logs')

# Cấu hình logging
log_filename = f'logs/tracking_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
# Khởi tạo tracker cho hình chữ nhật và hình tròn
tracker = MultiObjectTracker(
    dt=0.1, #Thời gian delta giữa các frame
    model_spec={
        'order_pos': 1,    # constant velocity model/Mô hình vận tốc không đổi
        'dim_pos': 2,      # 2D tracking
        'order_size': 1,   # constant size/ Kích thước không đổi
        'dim_size': 2,     # width, height/Chiều rộng và chiều cao
    },
    matching_fn_kwargs={
        'min_iou': 0.2 # Ngưỡng IoU tối thiểu để matching
    },
    active_tracks_kwargs={
        'min_steps_alive': 8,# Số frame tối thiểu để xác nhận track
        'max_staleness': 10,# Số frame tối đa cho phép mất dấu
    }
)

circle_id_counter = 1  # Biến để theo dõi ID cho hình tròn
rectangle_id_counter = 1  # Biến để theo dõi ID cho hình chữ nhật
circle_ids = {}  # Từ điển để lưu ID cho hình tròn
rectangle_ids = {}  # Từ điển để lưu ID cho hình chữ nhật

def detect_shapes(frame):
    logging.info("Starting shape detection")    
    # KHỬ NHIỄU
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Log preprocessing steps
    logging.info("Applying Gaussian Blur and converting to grayscale")
    blurred = cv2.GaussianBlur(frame, (11, 11), 3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    
    logging.info("Applying threshold and morphological operations")
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel1 = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    
    logging.info("Calculating Sobel gradients")
    grad_x = cv2.Sobel(closing, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(closing, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    logging.info("Finding contours")
    # Tìm contours
    contours, _ = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"Found {len(contours)} contours")
    detections = []
    circles = 0
    rectangles = 0
    for contour in contours:
        # Tính diện tích của contour
        area = cv2.contourArea(contour)
        
        # Lọc contour nhỏ
        if area < 1000:  # Điều chỉnh ngưỡng này theo nhu cầu
            continue
        # Tìm hình chữ nhật bao quanh
        x, y, w, h = cv2.boundingRect(contour)
        # Tính độ tròn
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        # Xấp xỉ đa giác
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # Phân loại hình dạng
        shape_type = None
        if len(approx) == 4 :  # Hình chữ nhật
            rectangles += 1
            shape_type = "rectangle"
        elif 0.8 <= circularity <= 1.1:  # Hình tròn
            circles += 1
            shape_type = "circle"
            
        
        
        if shape_type:
            # Tạo detection với class_id khác nhau cho mỗi loại hình
            detection = Detection(
                box=[x, y, x+w, y+h],  # [x1, y1, x2, y2]
                score=1.0,
                class_id=1 if shape_type == "circle" else 2  # 1: circle, 2: rectangle
            )
            detections.append(detection)
    logging.info(f"Detected {circles} circles and {rectangles} rectangles")       
    return detections    
def main():
    name = "Phan Thanh Thao"
    id = "22139062"
    logging.info("Starting tracking application")
    logging.info(f"Student Name: {name}")
    logging.info(f"Student ID: {id}")
    global circle_id_counter, rectangle_id_counter, circle_ids, rectangle_ids
    

    video_path = "daoquay.mp4"
    logging.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        logging.info(f"FAIL TO OPEN VIDEO:{video_path}")
        exit()
    
    frame_width = 640  # Độ rộng mong muốn
    frame_height = 320 
    logging.info(f"Video dimensions: {frame_width}x{frame_height}")

     # Khởi tạo VideoWriter
    output_file = 'output_video.mp4'
    logging.info(f"Initializing video writer: {output_file}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width * 2, frame_height * 2))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Không thể đọc frame.")
            logging.info(f"Cann't read frame:{frame_count}")
            break
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        logging.info(f"Processing frame {frame_count}")
        # 1. Khung video gốc
        original_frame = frame.copy()
        cv2.putText(original_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 2. Khung ảnh đã qua xử lý Gaussian Blur
        blurred_frame = cv2.GaussianBlur(frame, (11, 11), 3)
        bublurred_frame2=blurred_frame.copy()
        cv2.putText(blurred_frame, "Gaussian Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 3. Khung ảnh nhị phân
        gray = cv2.cvtColor(bublurred_frame2, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
        kernel1 = np.ones((7, 7), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
        binary_frame_color = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        cv2.putText(binary_frame_color, "Binary", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 4. Khung ảnh tracking
        detections = detect_shapes(frame)
        tracker.step(detections=detections)
        active_tracks = tracker.active_tracks()
        logging.info(f"Active tracks: {len(active_tracks)}")
        tracking_frame = frame.copy()
        for track in tracker.active_tracks():
            x1, y1, x2, y2 = map(int, track.box)
            if track.class_id == 1:  # Hình tròn
                if track.id not in circle_ids:
                    circle_ids[track.id] = circle_id_counter
                    circle_id_counter += 1
                track_id = circle_ids[track.id]
                color = (0, 255, 0)  # Xanh lá cho hình tròn
                shape_name = "Circle"
            else:  # Hình chữ nhật
                if track.id not in rectangle_ids:
                    rectangle_ids[track.id] = rectangle_id_counter
                    rectangle_id_counter += 1
                track_id = rectangle_ids[track.id]
                color = (0, 0, 255)  # Đỏ cho hình chữ nhật
                shape_name = "Rectangle"
            
            cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{shape_name} ID:{track_id}"
            cv2.putText(tracking_frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(tracking_frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ghép các khung hình lại với nhau
        top_row = cv2.hconcat([original_frame, blurred_frame])
        bottom_row = cv2.hconcat([binary_frame_color, tracking_frame])
        combined_frame = cv2.vconcat([top_row, bottom_row])
        
        # Thêm tên và ID
        cv2.putText(combined_frame, f"Name: {name} ID: {id}", (10, combined_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Combined Frames", combined_frame)
          # Ghi khung hình vào video
        out.write(combined_frame)
        if cv2.waitKey(20) & 0xFF == 27:  # Nhấn 'ESC' để thoát
            logging.info("ESC pressed, stopping application")
            break
    logging.info(f"Processed total {frame_count} frames")
    logging.info("Cleaning up resources")
    
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application finished successfully")
if __name__ == "__main__":
    main()