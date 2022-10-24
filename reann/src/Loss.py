import torch 
import torch.nn as nn

class Loss(nn.Module):
   def __init__(self):
      super(Loss, self).__init__()
      self.loss_fn=nn.MSELoss(reduction="sum")

   def forward(self,var,ab): 
      return  torch.cat([self.loss_fn(ivar,iab).view(-1) for ivar, iab in zip(var,ab)])

import torch as tc
class MSEFLoss(nn.Module):
   def __init__(self):
      super(MSEFLoss, self).__init__()
      # self.muE = muE
      # self.muF = muF

   def forward(self, var, ab, numatoms=None):

      y1 = [i for i in var]
      y0 = [i for i in ab]
      NC = len(y1[0])
      NTA = len(y1[1])

      ee = (y1[0] - y0[0])/numatoms
      ff = y1[1] - y0[1]
      self.lossE = tc.sum(ee*ee).view(-1)
      self.lossG = tc.sum(ff*ff).view(-1)
      return torch.cat([self.lossE/NC, self.lossG/NTA])