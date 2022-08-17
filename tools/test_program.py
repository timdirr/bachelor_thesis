import torch.nn as nn
import torch

x = torch.rand([32, 256, 7, 7])

x = x.mean(dim=[2, 3])

print(x.shape)