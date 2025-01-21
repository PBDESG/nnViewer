import torch
from torch import nn
import torch.nn.functional as F

function_to_wrap = [
    # torch.Tensor
    (torch.Tensor.__add__, torch.Tensor),
    (torch.Tensor.__div__, torch.Tensor),
    (torch.Tensor.__getitem__, torch.Tensor),
    (torch.Tensor.__iadd__, torch.Tensor),
    (torch.Tensor.__isub__, torch.Tensor),
    (torch.Tensor.__matmul__, torch.Tensor),
    (torch.Tensor.__mul__, torch.Tensor),
    (torch.Tensor.__rmul__, torch.Tensor),
    (torch.Tensor.__neg__, torch.Tensor),
    (torch.Tensor.__sub__, torch.Tensor),
    (torch.Tensor.__truediv__, torch.Tensor),
    (torch.Tensor.__pow__, torch.Tensor),
    (torch.Tensor.exp, torch.Tensor),
    (torch.Tensor.expand, torch.Tensor),
    (torch.Tensor.flatten, torch.Tensor),
    (torch.Tensor.mean, torch.Tensor),
    (torch.Tensor.neg, torch.Tensor),
    (torch.Tensor.neg_, torch.Tensor),
    (torch.Tensor.negative, torch.Tensor),
    (torch.Tensor.permute, torch.Tensor),
    (torch.Tensor.reshape, torch.Tensor),
    (torch.Tensor.t, torch.Tensor),
    (torch.Tensor.transpose, torch.Tensor),
    (torch.Tensor.view, torch.Tensor),

    # torch
    (torch.cat, torch),
    (torch.div, torch),
    (torch.exp, torch),
    (torch.exp_, torch),
    (torch.matmul, torch),
    (torch.pow, torch),
    (torch.stack, torch),
    (torch.sum, torch),
    (torch.neg, torch),
    (torch.t, torch),
    (torch.rsqrt, torch),
    (torch.sqrt, torch),
    (torch.sigmoid, torch),

    # torch.nn
    (nn.Conv2d, nn),
    (nn.Embedding, nn),

    # torch.nn.functional
    (F.conv1d, F),
    (F.conv2d, F),
    (F.embedding, F),
    (F.scaled_dot_product_attention, F),
    (F.interpolate, F),
]
