'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m  #0.2
        self.s = s  #30
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)  #[5994,192]
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)  #0.98006
        self.sin_m = math.sin(self.m)  #0.1986
        self.th = math.cos(math.pi - self.m)  #-0.98006
        self.mm = math.sin(math.pi - self.m) * self.m   #0.0397

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  #[400,5994]
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))  #[400,5994]
        phi = cosine * self.cos_m - sine * self.sin_m                     #[400,5994]
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s  #[400,5994]
        
        loss = self.ce(output, label)   #label [400]
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1