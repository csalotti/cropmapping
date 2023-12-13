from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """Focal Loss implementation

    Attributes:
        gamma (float) : Reduction factor
        alpha (Tensor[int|float]) : input weights
        avg (bool) : loss averaged flg, otherwise summd
    """

    def __init__(
        self,
        gamma: float = 0,
        alpha: Optional[int | float] = None,
        avg: bool = True,
    ):
        """Focal Loss Initialization

        Args:
            gamma (float) : Reduction factor
            alpha (Optional[int|float]) : input weights
            avg (bool) : loss averaged flg, otherwise summd

        """
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None
        self.avg = avg

    def forward(self, input: Tensor, target: Tensor):
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.avg:
            return loss.mean()
        else:
            return loss.sum()
