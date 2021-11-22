# from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import warnings
import PIL.Image
from torch.autograd import Variable

def hist_similar(input, target):
    input = torch.Tensor.float(input)
    target = torch.Tensor.float(target)
    # print('input ', input.type())
    # print('target ', target.type())
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    else:
        for l, r in zip(input, target):
            # print('l:', l)
            # print('r:', r)
            # print('torch.abs(l - r):', torch.abs(l - r))

            sum = Variable(torch.sum(1 - (0 if l == r else (torch.abs(l - r))) / torch.max(l, r)))

    # print('sum type ', sum.type()) #torch.cuda.LongTensor
    similar = sum / len(input)
    # print('similar type ', similar.type())
    return similar
#

def cal_similar(li, ri):

    return hist_similar(torch.histc(li), torch.histc(ri))