import torch
from torch import nn
from tools.utils import intersection_over_union
from tools.utils import convert_to_yolo


class YOLOLoss(nn.Module):
    """
    This class represent YOLOv1/v2 loss function.
    All calculations are based on tensors.
    """
    def __init__(self, anchors):
        """
        param: anchors - anchor boxes

        Note: be careful with predicted and target views.
        Expected target view: (batch size, s, s, num anchors, 5 + num classes)
        Epected predicted view: (batch size, s, s, num anchors * (5 + num classes))
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='none')

        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = self.anchors.size(0)

        # according to the YOLOv2 config file
        self.lamda_obj = 5
        self.lambda_noobj = 1
        self.lambda_coord = 1
        self.lambda_class = 1

    def forward(self, predictions, targets):
        s = predictions.size(1)
        batch_size = targets.size(0)
        num_classes = int((predictions.size(-1) - 5 * self.num_anchors) / 5)

        # convert predictions view to target view
        predictions = predictions.view(batch_size, s, s, self.num_anchors, 5 + num_classes)

        # target and predicted anchor box values:
        # target:     [conf, tx, ty, tw, th, c1, c2, ..., cn]
        # prediction: [conf, tx, ty, tw, th, c1, c2, ..., cn]

        predict_obj = torch.sigmoid(predictions[..., 0])
        predict_txty = torch.sigmoid(predictions[..., 1:3])
        predict_twth = predictions[..., 3:5]
        predict_classes = torch.softmax(predictions[..., 5:], dim=-1)

        target_obj = (targets[..., 0] == 1)
        target_noobj = (targets[..., 0] == 0)
        target_txty = targets[..., 1:3]
        target_twth = targets[..., 3:5]
        target_classes = targets[..., 5:]

        with torch.no_grad():
            # convert values from predicted format to standard YOLO format
            predicted_bbox = convert_to_yolo(predictions, self.anchors, s, target=False)
            target_bbox = convert_to_yolo(targets, self.anchors, s, target=True)

            # calculate iou between predictions and targets
            iou = intersection_over_union(predicted_bbox, target_bbox)

        # Note: loss are calculated only for targets with a confidence of 0 or 1
        #  1 = if object exist with the best anchor
        # -1 = if object exist with not the best anchor (iou > threshold)
        #  0 = others variants (object not exist)

        loss_xy = self.mse(predict_txty, target_txty).sum(dim=-1) * target_obj
        loss_xy = loss_xy.sum() / batch_size

        loss_wh = self.mse(torch.sqrt(predict_twth), torch.sqrt(target_twth)).sum(dim=-1) * target_obj
        loss_wh = loss_wh.sum() / batch_size

        loss_obj = self.mse(predict_obj, iou) * target_obj
        loss_obj = loss_obj.sum() / batch_size

        loss_no_obj = self.mse(predict_obj, iou * 0) * target_noobj
        loss_no_obj = loss_no_obj.sum() / batch_size

        loss_class = self.mse(predict_classes, target_classes).sum(dim=-1) * target_obj
        loss_class = loss_class.sum() / batch_size

        total_loss = self.lambda_coord * (loss_xy + loss_wh) + self.lamda_obj * loss_obj + \
            self.lambda_noobj * loss_no_obj + self.lambda_class * loss_class

        # expected total_loss propagation
        return [total_loss, loss_xy, loss_wh, loss_obj, loss_no_obj, loss_class]
