import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe

class GetACEDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list, orbital_cluster_expansion=None, **kwargs):
        super(GetACEDensity,self).__init__()
        '''
        Parameters
        ----------
        rs: tensor[ntype,nwave] float
        inta: tensor[ntype,nwave] float
        nipsin: np.array/list   int
        cutoff: float

        orbital_cluster_expansion: int
             atomic cluster expansion in the orbital direction. 
        '''
        self.rs=nn.parameter.Parameter(rs)
        self.inta=nn.parameter.Parameter(inta)
        self.register_buffer('orbital_cluster_expansion', torch.Tensor([orbital_cluster_expansion]))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.tensor([nipsin]))
        npara=[1]
        index_para=torch.tensor([0],dtype=torch.long)
        for i in range(1,nipsin):
            npara.append(np.power(3,i))
            index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i))
        norbkind = nipsin
        if orbital_cluster_expansion != 0:
            # ACE index
            ace_idx = []
            for i in range(orbital_cluster_expansion):
                norbkind += 1
                ace_idx.extend([i + nipsin]*(3**i))
            index_para=torch.cat((index_para, torch.tensor(ace_idx).long()))
        ntype, nwave = rs.shape
        NO = len(index_para)
        self.register_buffer('index_para',index_para)
        self.register_buffer('NO',torch.Tensor([NO]).long())
        self.register_buffer('ntype',torch.Tensor([ntype]).long())
        self.register_buffer('nwave',torch.Tensor([nwave]).long())
        self.register_buffer('norbit',torch.Tensor([norbit]).long())
        self.register_buffer('norbkind',torch.Tensor([norbkind]).long())
        # NS x nmax
        # ntype x nmax
        self.params=nn.parameter.Parameter(torch.ones_like(self.rs)/float(neigh_atoms))
        #self.params=nn.parameter.Parameter(torch.ones(ntype, NO, nwave)/float(neigh_atoms))
        # loop x (lmax+3) x nmax x norbit
        self.hyper=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.rand(self.rs.shape[1],norbit)).\
        unsqueeze(0).unsqueeze(0).repeat(len(ocmod_list)+1, norbkind,1,1))
        ocmod=OrderedDict()
        for i, m in enumerate(ocmod_list):
            f_oc="memssage_"+str(i)
            ocmod[f_oc]= m
        self.ocmod= torch.nn.ModuleDict(ocmod)

        #self.wcc0 = torch.nn.Linear(2*nwave, nwave)
        #self.wcc1 = torch.nn.Linear(4*nwave, nwave, bias=False)
        #self.wcc2 = torch.nn.Linear(3*nwave, nwave, bias=False)

    def gaussian(self,distances,species_):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        distances=distances.view(-1,1)
        radial=torch.empty((distances.shape[0],self.rs.shape[1]),dtype=distances.dtype,device=distances.device)
        for itype in range(self.rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0]>0:
                part_radial=torch.exp(self.inta[itype:itype+1]*torch.square \
                (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))
                radial.masked_scatter_(mask.view(-1,1),part_radial)
        return radial
    
    def cutoff_cosine(self,distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*numatom*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)
    
    def angular(self,dist_vec):
        # Tensor: dist_vec[neighbour*numatom*nbatch,3]
        # return: angular[neighbour*numatom*nbatch,npara[0]+npara[1]+...+npara[ipsin]]
        NN, _ = dist_vec.shape
        dtype, device = dist_vec.dtype, dist_vec.device
        totneighbour=dist_vec.shape[0]
        # NNx3 -> 3 x NN
        dist_vec=dist_vec.permute(1,0).contiguous()
        # 1xNN
        # orbital=f_cut.view(1,-1)
        orbital = torch.ones((1, NN), dtype=dtype, device=device) 
        # NO x NN
        angular=torch.empty(self.index_para.shape[0],totneighbour,dtype=dtype,device=device)
        angular[0]= 1.
        num=1
        for ipsin in range(1,self.nipsin[0]):
            # NOxNN x 3xNN -> NOx3xNN -> 3NO x NN
            orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        
        if self.orbital_cluster_expansion[0] != 0:
            ace_l = self.orbital_cluster_expansion[0]
            # 000 case
            if ace_l > 0:
                angular[num] = 1.
                num += 1
            # 100 case
            if ace_l > 1:
                angular[num:num+3] = angular[1:4]
                num += 3
            # 110 case
            if ace_l > 2:
                # NOxNN x 3xNN -> NOx3xNN -> NO3 x NN
                orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
                # orbital = (orbital[:, None, :] * dist_vec[None]).reshape(-1, NN)
                angular[num:num+9] = orbital
                num += 9
            # 111 case
            if ace_l > 3:
                # NOxNN x 3xNN -> NOx3xNN -> NO3 x NN
                orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
                # orbital = (orbital[:, None, :] * dist_vec[None]).reshape(-1, NN)
                angular[num:num+27] = orbital
                num += 27
        
        return angular  
    
    def forward(self,cart,numatoms,species,atom_index,shifts):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom

        attribute
        ---------
        NO : int, Number of orbitals.
        """
        NO, nwave, norbit = self.NO.item(), self.nwave.item(), self.norbit.item()
        tmp_index=torch.arange(numatoms.shape[0],device=cart.device)*cart.shape[1]
        self_mol_index=tmp_index.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)
        cart_=cart.flatten(0,1)
        totnatom=cart_.shape[0]
        padding_mask=torch.nonzero((shifts.view(-1,3)>-1e10).all(1)).view(-1)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12=(atom_index.view(2,-1)+self_mol_index)[:,padding_mask]
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values=shifts.view(-1,3).index_select(0,padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec,dim=-1)
        dist_vec=dist_vec/distances.view(-1,1)
        # NTA -> NN
        species_ = species.index_select(0,atom_index12[1])
        # NTA -> NTA x NO
        # extended_species = species[:, None].expand(totnatom, NO).reshape(totnatom*NO)

        # NN
        cutoff = self.cutoff_cosine(distances)
        # print(torch.linalg.norm(cutoff))
        # NNxnmax -> NNxNOxnmax
        radial = self.radial_ace_expand(self.gaussian(distances,species_))# * cutoff.view(-1, 1))
        # NOxNN -> NNxNO
        angular = self.angular(dist_vec).permute(1,0).contiguous()
        # NOxNN x NNxnmax -> NNxNOx(nmax)
        # orbital = oe.contract("ji,ik -> ijk", angular, radial,backend="torch")
        # NNxNO x NNxNOxnmax -> NNxNOx(nmax)
        orbital = oe.contract("ij,ijk -> ijk", angular, radial,backend="torch")
        # NTA x nmax
        orb_coeff=torch.empty((totnatom,self.rs.shape[1]),dtype=cart.dtype,device=cart.device)
        # orb_coeff=torch.empty((totnatom, NO, self.rs.shape[1]),dtype=cart.dtype,device=cart.device)
        mask=(species>-0.5).view(-1)
        # NS x nmax -> NTA x nmax
        orb_coeff.masked_scatter_(mask.view(-1,1),self.params.index_select(0,species[torch.nonzero(mask).view(-1)]))
        # NS x NO x nmax -> NTA x NO x nmax
        #orb_coeff.masked_scatter_(mask.view(-1,1,1),self.params.index_select(0,species[torch.nonzero(mask).view(-1)]))
        # NTA x (norb)
        density=self.obtain_orb_coeff(0,totnatom,orbital,atom_index12,orb_coeff).view(totnatom,-1)
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            # NTA x nmax | m: NTAx(norb), NTA -> NTA x nmax
            # species: NTA x NO
            # orb_coeff = orb_coeff + m(density.reshape(totnatom*NO, norbit), extended_species).reshape(totnatom, NO, nwave)
            orb_coeff = orb_coeff + m(density, species)
            # NTA x (norb)
            density = self.obtain_orb_coeff(ioc_loop+1,totnatom,orbital,atom_index12,orb_coeff)
 
        # NTA x NO x (norb) -> NTA x (norb)
        # return torch.sum(density, dim=1)
        return density
 
    def obtain_orb_coeff(self,iteration:int,totnatom:int,orbital,atom_index12,orb_coeff):
        # NTAx(nmax) -> NNx(nmax)
        # NTAxNOx(nmax) -> NNxNOx(nmax)
        expandpara=orb_coeff.index_select(0,atom_index12[1])

        # NNxNOx(nmax) x NNx(nmax)
        # NNxNOx(nmax) x NNx(NO)x(nmax)
        # TODO: Tensor product this 
        worbital=oe.contract("ijk,ik->ijk", orbital,expandpara,backend="torch")
        # worbital = self.tensor_prod(orbital, expandpara)
        # NTA x NO x nmax
        sum_worbital=torch.zeros((totnatom,orbital.shape[1],self.rs.shape[1]),dtype=orbital.dtype,device=orbital.device)
        # NN x NO x nmax -> NTA x NO x nmax
        sum_worbital=torch.index_add(sum_worbital,0,atom_index12[0],worbital)
        # lmax x nmax x norbit -> NO x nmax x norbit
        expandpara=self.hyper[iteration].index_select(0,self.index_para)

        # NTAxNOx(nmax) x  NOx(nmax)x(norbit) -> NTAxNOx(norbit)
        hyper_worbital=oe.contract("ijk,jkm -> ijm",sum_worbital,expandpara,backend="torch")
        
        # NTA x NO x norbit -> NTA x norbit
        return torch.sum(torch.square(hyper_worbital), dim=1)

    def radial_ace_expand(self, r):
        """ Calculate the orbital components before conduct summation & squre
        r: NN x nmax

        Return
        ------

        NNxNOxnmax
        """
        NN, nmax = r.shape
        NO = self.NO

        if self.orbital_cluster_expansion[0] != 0:
            orbit = (3**(self.nipsin[0]) - 1) //2
            ace_orbit = (3**(self.norbkind - self.nipsin[0]) - 1) //2
            ace_r = torch.sqrt(r) ** 3
            return torch.cat([r[:, None, :].expand(NN, orbit, nmax), ace_r[:, None, :].expand(NN, ace_orbit, nmax)], dim=1)
        
        # Boradcast: NNxnmax -> NNxNOxnmax
        return r[:, None, :].expand(NN, NO, nmax)


    def tensor_prod(self, a, b):
        """
        a: orbital : NNxNOx(nmax)
        b: expandpara : NNxNOx(nmax)

        """
        NN, NO, nmax = a.shape
        tp000 = a[:, 0, :] * b[:, 0, :]
        tp110 = oe.contract("ijk, ijk -> ik", b[:, 1:4, :], b[:, 1:4, :], backend="torch")
        temp = torch.cat((tp000, tp110), dim=1)
        tp0 = self.Wcc0(temp)

        tp011 = a[:, 0, None, :] * b[:, 1:4, :]
        tp101 = a[:, 1:4, :] * b[:, 0,None, :]
        tp211 = self.tp121(a[:, 4:, :], b[:, 1:4, :])
        tp121 = self.tp121(b[:, 4:, :], a[:, 1:4, :])
        tp1 = self.Wcc1(torch.cat((tp011, tp101, tp211, tp121), dim=2))

        tp112 = oe.contract("ijl, ikl->ijkl", a[:, 1:4, :], b[:, 1:4, :]).reshape(NN, 9, nmax)
        tp202 = a[:, 4:, :] * b[:, 0, None, :]
        tp022 = a[:, 0, None, :] * b[:, 4:, :]
        tp2 = self.Wcc2(torch.cat((tp112, tp202, tp022), dim=2))
        return torch.cat((tp0, tp1, tp2), dim=1)

    def tp121(self, a, b):
        # NN x 1 x (2 * nmax) -> NN x 1 x nmax
        NN, _, nmax = a.shape
        rank2 = a.reshape(NN, 3, 3, nmax)
        # ! check
        return oe.contract("ijkl, ikl -> ijl", rank2, b)

    def Wcc0(self, a):
        # NN x 1 x (2 * nmax) -> NN x 1 x nmax
        NN, nmax2 = a.shape
        return self.wcc0(a.reshape(NN, nmax2)).reshape(NN, 1, nmax2//2)

    def Wcc1(self, a):
        # NN x 3 x (4 * nmax) -> NN x 3 x nmax
        NN, _, nmax4 = a.shape
        return self.wcc1(a.reshape(NN*3, nmax4)).reshape(NN, 3, nmax4//4)

    def Wcc2(self, a):
        # NN x 9 x (3 * nmax) -> NN x 9 x nmax
        NN, _, nmax3 = a.shape
        return self.wcc2(a.reshape(NN*9, nmax3)).reshape(NN, 9, nmax3//3)


