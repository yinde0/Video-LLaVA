import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizeVideo(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1,1,1,1))
        self.register_buffer("std",  torch.tensor(std).view(-1,1,1,1))
    def forward(self, x):  # x: (C,T,H,W) in [0,1]
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

class ShortSideScale(nn.Module):
    def __init__(self, size:int):
        super().__init__()
        self.size = int(size)
    def forward(self, x):  # (C,T,H,W)
        C,T,H,W = x.shape
        short = min(H,W)
        if short == self.size: return x
        scale = self.size / short
        new_h, new_w = int(round(H*scale)), int(round(W*scale))
        xt = x.permute(1,0,2,3)  # (T,C,H,W)
        xt = F.interpolate(xt, size=(new_h,new_w), mode="bilinear", align_corners=False)
        return xt.permute(1,0,2,3)

class CenterCropVideo(nn.Module):
    def __init__(self, size:int):
        super().__init__()
        self.size = int(size)
    def forward(self, x):  # (C,T,H,W)
        th = tw = self.size
        i = max((x.shape[2]-th)//2, 0)
        j = max((x.shape[3]-tw)//2, 0)
        return x[..., i:i+th, j:j+tw]

class RandomHorizontalFlipVideo(nn.Module):
    def __init__(self, p:float=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=[3])  # flip width
        return x
