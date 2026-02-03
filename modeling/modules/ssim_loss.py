import torch
import torch.nn as nn
import pyiqa


class SSIMLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self._device = device
        self._criterion = None

    def _get_criterion(self, device):
        if self._criterion is None or self._device != device:
            self._criterion = pyiqa.create_metric("ssimc", device=device, as_loss=True)
            self._device = device
        return self._criterion

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # SSIM expects 1 or 3 channels. Reduce other channel counts to 1.
        if pred.shape[1] not in (1, 3):
            if pred.shape[1] >= 7:
                pred = pred[:, [0, 2, 6], ...]
            else:
                pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] not in (1, 3):
            if target.shape[1] >= 7:
                target = target[:, [0, 2, 6], ...]
            else:
                target = target.mean(dim=1, keepdim=True)
        criterion = self._get_criterion(pred.device)
        return 1 - criterion(pred, target)
