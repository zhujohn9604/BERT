import torch
import math


def mish(x):
    #  Mish: https: // arxiv.org / pdf / 1908.08681.pdf
    return x * torch.tanh(torch.log(1 + torch.exp(x)))


def gelu(x):
    return 0.5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3.))))


def swish(x):
    return x * torch.sigmoid(x)
