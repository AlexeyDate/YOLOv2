import torch
from tools.utils import intersection_over_union


def mean_average_precision(pred_boxes, true_boxes, classes, iou_threshold=0.5) -> float:
    """
    Mean Average Precision of bounding boxes.

    param: pred_boxes - predicted bounding boxes
    param: true_boxes - true bounding boxes
    param: classes - number of classes
    param: iou_threshold - threshold of IOU (default = 0.5)

    return: calculated mAP
    """

    average_precisions = []

    for current_class in range(classes):
        FP = 0
        FN = 0
        TP = 0
        precisions = []
        recalls = []

        for i in range(len(true_boxes)):
            FN += len([box for box in true_boxes[i] if torch.argmax(box[5:]) == current_class])

        for i in range(len(pred_boxes)):
            pred_boxes_class = [box for box in pred_boxes[i] if torch.argmax(box[5:]) == current_class]
            true_boxes_class = [box for box in true_boxes[i] if torch.argmax(box[5:]) == current_class]
            for k in range(len(pred_boxes_class)):
                max_iou = 0
                max_index = 0
                for j in range(len(true_boxes_class)):
                    if intersection_over_union(pred_boxes_class[k], true_boxes_class[j]).item() > max_iou:
                        max_iou = intersection_over_union(pred_boxes_class[k], true_boxes_class[j]).item()
                        max_index = j

                if max_iou < iou_threshold:
                    FP += 1
                else:
                    TP += 1
                    FN -= 1
                    true_boxes_class.pop(max_index)

                precisions.append(TP / (TP + FP))
                recalls.append(TP / (TP + FN))

        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        average_precisions.append(torch.trapezoid(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
