import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmodlist, cluster_number=1):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmodlist[0]
        self.cluster_number = cluster_number
        print('Property_force_ace', cluster_number)
        if len(nnmodlist) > 1:
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]

    def forward(self,cart,numatoms,species,atom_index,shifts,create_graph=True):
        cart.requires_grad=True
        species=species.view(-1)
        # NTA x NO x norbit
        density = self.density(cart,numatoms,species,atom_index,shifts)
        # NTA, NO, norbit = density.shape
        # NTAxNOx(norbit) -> NTA x NO(norbit)
        NTA = len(density)
        density = density.reshape(NTA, -1)
        descriptor = [density]
        for k in range(1, self.cluster_number):
            temp = oe.contract('ij,ik -> ijk', descriptor[-1], density, backend='torch').reshape(NTA, -1)
            descriptor.append(temp)
        density = torch.cat(descriptor, axis=1)
        # NTA x norbit^cluster_number
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        grad_outputs=torch.ones(numatoms.shape[0],device=cart.device)
        # NTA x 3
        force=-torch.autograd.grad(varene,cart,grad_outputs=grad_outputs,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0].view(numatoms.shape[0],-1)
        return varene,force

