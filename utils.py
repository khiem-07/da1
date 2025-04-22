# utils.py
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

def get_metrics(num_classes, device):
    """
    Trả về metric: IoU và Accuracy.
    """
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    return iou_metric, accuracy_metric
