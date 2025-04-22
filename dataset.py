# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform_img=None):
        """
        Args:
            image_dir (str): Đường dẫn tới thư mục chứa ảnh input.
            label_dir (str): Đường dẫn tới thư mục chứa ảnh label.
            transform_img (callable, optional): Các transform áp dụng cho ảnh input.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform_img = transform_img
        # Dictionary chuyển đổi màu thành nhãn
        self.class_colors = {
            (2, 0, 0): 0,       
            (127, 0, 0): 1,     
            (248, 163, 191): 2  
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        # Đọc ảnh input (RGB)
        image = Image.open(img_path).convert("RGB")
        # Đọc ảnh label dưới dạng RGB để lấy mã màu chính xác
        label_img = Image.open(label_path).convert("RGB")

        if self.transform_img:
            image = self.transform_img(image)
        else:
            image = T.ToTensor()(image)

        # Chuyển đổi label từ ảnh RGB sang mảng nhãn (values từ 0 đến 2)
        label_np = np.array(label_img)
        # Khởi tạo mask với cùng kích thước (h, w)
        label_mask = np.zeros(label_np.shape[:2], dtype=np.uint8)
        for rgb, class_idx in self.class_colors.items():
            # So sánh từng kênh để xác định các pixel có màu giống như key
            matches = np.all(label_np == np.array(rgb), axis=-1)
            label_mask[matches] = class_idx

        label_tensor = torch.from_numpy(label_mask).long()
        return image, label_tensor

def get_loaders(image_dir, label_dir, batch_size=16):
    """
    Tạo DataLoader cho training và validation với tỉ lệ 80/20.
    """
    transform_img = T.Compose([
        T.ToTensor(),  # Chuyển ảnh input về tensor dạng [C, H, W]
    ])
    
    dataset = SegmentationDataset(image_dir, label_dir, transform_img=transform_img)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
