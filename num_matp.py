import cv2
from cv2 import aruco

# Tạo dictionary marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Khởi tạo ảnh trắng
img = cv2.UMat(200, 200, cv2.CV_8UC1)  # UMat hoạt động tốt hơn với GPU nhưng cũng tương thích CPU

# Vẽ marker ArUco ID = 0 vào ảnh
aruco_dict.generateImageMarker(0, 200, img)

# Chuyển UMat về NumPy array để lưu file
cv2.imwrite("aruco_marker_id0.png", img.get())
print("✅ Đã tạo mã ArUco marker (ID = 0) và lưu vào file aruco_marker_id0.png")
