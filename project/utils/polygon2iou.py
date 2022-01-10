import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import torchmetrics


def polygon_iou(pred_polygons: torch.Tensor, gt_masks: torch.Tensor):
    """COmputes the mIoU of the segmentation mask created from the predicted polygon
    """
    pred_polygons = pred_polygons.clone().detach().cpu().numpy()
    H, W = gt_masks.shape[1], gt_masks.shape[2]

    # create binary mask from polygon
    # denormalize the coordinates of the contours
    pred_polygons[:, :, 0] = pred_polygons[:, :, 0] * W
    pred_polygons[:, :, 1] = pred_polygons[:, :, 1] * H

    pred_masks = torch.zeros_like(gt_masks)
    i = 0
    for pred in pred_polygons:
        # unsqueeze to have the good format for cv2.drawcontours
        pred = np.expand_dims(pred, axis=1)
        # obtain mask from polygon
        pred_mask = cv2.drawContours(np.zeros_like(gt_masks, dtype=np.uint8), [pred], -1, 1, cv2.FILLED)

        pred_masks[i, :, :] = torch.Tensor(pred_mask)
        
        i += 1
        
    return torchmetrics.functional.iou(pred_masks, gt_masks)
