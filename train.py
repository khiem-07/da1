# train.py
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_loaders
from model import TinySegNet, count_parameters
from utils import get_metrics

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    iou_metric, acc_metric = get_metrics(num_classes, device)
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        iou_metric.update(preds, labels)
        acc_metric.update(preds, labels)
        pbar.set_postfix({'Loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_iou = iou_metric.compute().item()
    mean_acc = acc_metric.compute().item()
    return epoch_loss, mean_iou, mean_acc

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    iou_metric, acc_metric = get_metrics(num_classes, device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, labels)
            acc_metric.update(preds, labels)
            pbar.set_postfix({'Loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_iou = iou_metric.compute().item()
    mean_acc = acc_metric.compute().item()
    return epoch_loss, mean_iou, mean_acc

def main():
    # Thiết lập tham số
    image_dir = "images/input"  # Thư mục ảnh input
    label_dir = "images/label"  # Thư mục ảnh label
    batch_size = 16
    num_classes = 3
    num_epochs = 30
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lấy DataLoader
    train_loader, val_loader = get_loaders(image_dir, label_dir, batch_size)
    
    # Khởi tạo mô hình
    model = TinySegNet(n_classes=num_classes)
    print("Total parameters:", count_parameters(model))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_iou = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_iou, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_iou, val_acc = evaluate(model, val_loader, criterion, device, num_classes)

        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_model_state = copy.deepcopy(model.state_dict())
            print("  --> Best model updated!")
    
    # Lưu model tốt nhất bằng TorchScript
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, "Best_SegmentationModel.pt")
        print("Best model saved to Best_SegmentationModel.pt")
    
if __name__ == '__main__':
    main()
