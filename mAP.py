import torch
from torchvision.ops import box_iou
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def xywh_to_xyxy(boxes, w=480, h=640):
    x, y, w, h = boxes[:, 0]*w, boxes[:, 1]*h, boxes[:, 2]*w, boxes[:, 3]*h
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

# Example input in [x, y, w, h] format
boxes1 = torch.tensor([
    [0.521895, 0.501549, 0.224191, 0.669153],
])

# boxes2 = torch.tensor([
#     [0.521995, 0.501649, 0.225191, 0.670153],
# ]) # PointNet

boxes2 = torch.tensor([
    [0.524895, 0.504549, 0.227191, 0.666153],
]) # P4

# Convert to [x_min, y_min, x_max, y_max] format
boxes1_xyxy = xywh_to_xyxy(boxes1)
boxes2_xyxy = xywh_to_xyxy(boxes2)

# Calculate IoU
iou = box_iou(boxes1_xyxy, boxes2_xyxy)
print(iou)


all_precisions = []
all_recalls = []
all_ap = []

for class_id in range(1):
    preds = [p for p in boxes2]
    gts = [g for g in boxes1]

    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))
    detected = []

    for i, pred in enumerate(preds):
        ious = [0.9929,0.95]
        max_iou = max(ious) if ious else 0
        max_iou_index = np.argmax(ious) if ious else -1

        if max_iou >= 0.7 and max_iou_index not in detected:
            TP[i] = 1
            detected.append(max_iou_index)
        else:
            FP[i] = 1

    cumulative_TP = np.cumsum(TP)
    cumulative_FP = np.cumsum(FP)

    precision = cumulative_TP / (cumulative_TP + cumulative_FP)
    recall = cumulative_TP / len(gts)

    all_precisions.append(precision)
    all_recalls.append(recall)

    ap = average_precision_score(TP, precision)
    all_ap.append(ap)

mAP = np.mean(all_ap)
print(mAP)