import torch
import numpy as np
from torch import nn

def dice_coef(y_pred, y_true):

    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    y_pred = torch.where(y_pred > 0.5, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, ones, zeros)
    
    if torch.cuda.is_available():
        y_true = y_true.cuda()


    # Calculate Dice Coefficient Score
    smooth = 1.
    
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()

    return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def dice_coef_np(y_pred, y_true):
    

    # Convert to Binary
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    # Calculate Dice Coefficient Score
    smooth = 1.
    
    y_pred = y_pred.reshape((-1))
    y_true = y_true.reshape((-1))

    intersection = np.sum(np.matmul(y_pred, y_true))

    return (2 * intersection + smooth) / (np.sum(y_pred) + np.sum(y_true) + smooth)


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"


    # Convert to Binary
    zeros = torch.zeros(input.size())
    ones = torch.ones(input.size())

    target = target.cpu()
    target = torch.where(target > 0.5, ones, zeros)

    if torch.cuda.is_available():
        target = target.cuda()

    input = input.cpu()
    input = torch.where(input > 0.5, ones, zeros)
    
    if torch.cuda.is_available():
        input = input.cuda()

    input = flatten(input)
    target = flatten(target)

    # target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    denominator = (input + target).sum(-1)

    return (2. * intersect + epsilon) / (denominator + epsilon)


class DiceCoef(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, sigmoid_normalization=True, return_score_per_channel=False):
        super(DiceCoef, self).__init__()
        self.epsilon = epsilon
        self.return_score_per_channel = return_score_per_channel
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target

    def forward(self, input, target):
        input = self.normalization(input)
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon)
        # Average the Dice score across all channels/classes
        if self.return_score_per_channel:
            return per_channel_dice
        else:
            return torch.mean(per_channel_dice)
            
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
