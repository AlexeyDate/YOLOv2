import torch
from tqdm import tqdm


def intersection_over_union(predicted_bbox, ground_truth_bbox) -> torch.tensor:
    """
    Intersection Over Union for 2 rectangles.

    param: predicted_bbox - predicted tensor
    param: ground_truth_bbox - target tensor

    return: Intersection Over Union tensor

    Note: be careful with tensor views.
    Epected predicted_bbox view: (batch size, s, s, num anchors, (5 + num classes)
    Expected ground_truth_bbox view: (batch size, s, s, num anchors, 5 + num classes)
    """

    # predicted and target values (standard YOLO format):
    # target:     [conf, x, y, w, h, c1, c2, ..., cn]
    # prediction: [conf, x, y, w, h, c1, c2, ..., cn]

    # convert values to x1, y1, x2, y2
    predicted_bbox_x1 = predicted_bbox[..., 1] - predicted_bbox[..., 3] / 2
    predicted_bbox_x2 = predicted_bbox[..., 1] + predicted_bbox[..., 3] / 2
    predicted_bbox_y1 = predicted_bbox[..., 2] - predicted_bbox[..., 4] / 2
    predicted_bbox_y2 = predicted_bbox[..., 2] + predicted_bbox[..., 4] / 2

    ground_truth_bbox_x1 = ground_truth_bbox[..., 1] - ground_truth_bbox[..., 3] / 2
    ground_truth_bbox_x2 = ground_truth_bbox[..., 1] + ground_truth_bbox[..., 3] / 2
    ground_truth_bbox_y1 = ground_truth_bbox[..., 2] - ground_truth_bbox[..., 4] / 2
    ground_truth_bbox_y2 = ground_truth_bbox[..., 2] + ground_truth_bbox[..., 4] / 2

    intersection_x1 = torch.max(predicted_bbox_x1, ground_truth_bbox_x1)
    intersection_x2 = torch.min(predicted_bbox_x2, ground_truth_bbox_x2)
    intersection_y1 = torch.max(predicted_bbox_y1, ground_truth_bbox_y1)
    intersection_y2 = torch.min(predicted_bbox_y2, ground_truth_bbox_y2)

    zeros = torch.zeros_like(predicted_bbox_x1)
    intersection_area = torch.max(intersection_x2 - intersection_x1, zeros) * torch.max(
        intersection_y2 - intersection_y1, zeros
    )

    area_predicted = (predicted_bbox_x2 - predicted_bbox_x1) * (predicted_bbox_y2 - predicted_bbox_y1)
    area_gt = (ground_truth_bbox_x2 - ground_truth_bbox_x1) * (ground_truth_bbox_y2 - ground_truth_bbox_y1)

    union_area = area_predicted + area_gt - intersection_area

    iou = intersection_area / union_area
    return iou


def inersection_over_union_anchors(bbox_wh, anchors) -> torch.tensor:
    """
    Intersection Over Union for boundig box with anchor boxes with same centers.

    param: bbox_wh - bounding box width and height
    param: anchors - anchor box width and height

    return: Intersection Over Union tensor
    """
    w1, h1 = bbox_wh
    num_anchors = anchors.size(0)
    iou = torch.empty(num_anchors, dtype=torch.float32)
    for i in range(num_anchors):
        w2, h2 = anchors[i]
        intersection_area = min(w1, w2) * min(h1, h2)
        union_area = (w1 * h1) + (w2 * h2) - intersection_area
        iou[i] = intersection_area / union_area

    return iou


def non_max_supression(bboxes, iou_threshold):
    """
    Non-Maximum Supression.

    param: bboxes - all predicted bounding boxes with valid confidience values
    param: iou_threshold - intesection over union threshold

    return: correct bounding boxes
    """
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    non_max_bboxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        non_max_bboxes.append(current_box)

        temp_bboxes = []
        for box in bboxes:
            class_box = torch.argmax(box[5:])
            class_current_box = torch.argmax(current_box[5:])

            if intersection_over_union(current_box, box).item() < iou_threshold or class_box != class_current_box:
                temp_bboxes.append(box)
        bboxes = temp_bboxes

    return non_max_bboxes


def convert_to_yolo(bbox, anchors, s, target=False) -> torch.Tensor:
    """
    convert predicted coordinates to standard YOLO format

    param: bbox - bounding boxes
    param: anchors - anchor boxes
    param: s - current grid size
    param: target - to calculate confidience using sigmoid function and
    classes using softmax function for predicted values (default=False).
    This is only for predictions.

    return: calculated all bounding boxes in standard YOLO format

    Note: tx, ty relative to grid cell convert to relative to image.
    tw, th convert to bw, bh by using anchor boxes
    """

    device = bbox.device
    anchors = anchors.to(device)

    grid_y, grid_x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    grid_y = grid_y.contiguous().view(1, s, s, 1).to(device)
    grid_x = grid_x.contiguous().view(1, s, s, 1).to(device)
 
    bbox[..., 1] = (torch.sigmoid(bbox[..., 1]) + grid_x) / s
    bbox[..., 2] = (torch.sigmoid(bbox[..., 2]) + grid_y) / s
    bbox[..., 3] = anchors[:, 0] * torch.exp(bbox[..., 3])
    bbox[..., 4] = anchors[:, 1] * torch.exp(bbox[..., 4])

    if not target:
        bbox[..., 0] = torch.sigmoid(bbox[..., 0])
        bbox[..., 5:] = torch.softmax(bbox[..., 5:], dim=-1)

    return bbox


def get_bound_boxes(loader, model, anchors, iou_threshold=0.5, threshold=0.4):
    """
    Getting predicted and target bounding boxes with Non-Maximum Supression.

    param: loader - dataloader
    param: model - model
    param: iou_threshold - Intersection Over Union threshold (default = 0.5)
    param: threshold - confidience threshold (default = 0.4)

    return: all prediction bounding boxes, all true bounding boxes
    """

    device = model.device

    assert isinstance(loader, torch.utils.data.dataloader.DataLoader),\
        "loader does not match the type of torch.utils.data.dataloader.DataLoader"

    model.eval()
    for i, batch in enumerate(tqdm(loader, desc=f'Prediction all bounding boxes', leave=False)):
        images = batch['image'].to(device)
        if i == 0:
            targets = batch['target'].to(device)
            with torch.no_grad():
                predictions = model(images)
        else:
            target = batch['target'].to(device)
            targets = torch.cat((targets, target))
            with torch.no_grad():
                predictions = torch.cat((predictions, model(images)))

    # setting values
    anchors = torch.tensor(anchors, dtype=torch.float32)
    num_anchors = anchors.size(0)
    s = predictions.size(1)
    size = predictions.size(0)
    num_classes = int((predictions.size(-1) - 5 * num_anchors) / 5)
    predictions = predictions.view(size, s, s, num_anchors, 5 + num_classes)

    # convert predictions and targets to standard YOLO format
    predicted_bbox = convert_to_yolo(predictions, anchors, s, target=False)
    target_bbox = convert_to_yolo(targets, anchors, s, target=True)

    all_pred_boxes = []
    all_true_boxes = []
    for i in range(size):
        mask = target_bbox[i, ..., 0] == 1
        image_true_boxes = target_bbox[i, mask, :]
        all_true_boxes.append(image_true_boxes)       

        mask = predicted_bbox[i, ..., 0] >= threshold
        image_pred_boxes = predicted_bbox[i, mask, :]
        image_pred_boxes = non_max_supression(image_pred_boxes, iou_threshold)
        all_pred_boxes.append(image_pred_boxes)

    return all_pred_boxes, all_true_boxes



