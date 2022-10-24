import torch 
import torch.nn as nn

class Loss2(nn.Module):
   def __init__(self):
      super(Loss, self).__init__()
      self.loss_fn=nn.MSELoss(reduction="sum")

   def forward(self,var,ab): 
      return  torch.cat([self.loss_fn(ivar,iab).view(-1) for ivar, iab in zip(var,ab)])

class Loss(nn.Module):
   def __init__(self):
      super(Loss2, self).__init__()
      
   def forward(self, var, ab):
      self.record = []
      errs = []
      for ivar, iab in zip(var, ab):
         error = (ivar - iab)*(ivar - iab)
         self.record.append(error.cpu().detach())
         errs.append(torch.sum(error))
      return torch.cat(errs)
      