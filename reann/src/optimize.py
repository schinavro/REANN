import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def Optimize(fout,prop_ceff,nprop,train_nele,test_nele,init_f,final_f,decay_factor,start_lr,end_lr,print_epoch,Epoch,\
data_train,data_test,Prop_class,loss_fn,optim,scheduler,ema,restart,PES_Normal,device,PES_Lammps=None): 

    rank=dist.get_rank()
    best_loss=1e30*torch.ones(1,device=device)    

    optim.param_groups[0]["lr"] = start_lr

    fout.write("{0:<5s} {1:>8s} {2:5s} {3:>16s} {4:>16s} {5:>5s} {6:>16s} {7:>16s} \n".format(
                "Epoch", "lr", "Train", "RMSE-E(meV/atom)", "RMSE-F(meV/å)", " Test", "RMSE-E(meV/atom)", "RMSE-F(meV/atom)"))
    for iepoch in range(Epoch): 
        # set the model to train
       Prop_class.train()
       lossprop=torch.zeros(nprop+2,device=device)        
       # NC, NTA, lossE, lossG = 0, 0, 0, 0
       for data in data_train:
          abProp,cart,numatoms,species,atom_index,shifts=data
          loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts),abProp, numatoms)
          loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
          # clear the gradients of param
          #optim.zero_grad()
          optim.zero_grad(set_to_none=True)
          #print(torch.cuda.memory_allocated)
          # obtain the gradients
          loss.backward()
          optim.step()   

          if np.mod(iepoch,print_epoch)==0:
              lossprop[0] += len(numatoms)
              lossprop[1] += torch.sum(numatoms)
              lossprop[2] += loss_fn.lossE.item()
              lossprop[3] += loss_fn.lossG.item()

          # lossprop += loss_fn.detach()
          #doing the exponential moving average update the EMA parameters
          #ema.update()
    
       #  print the error of vailadation and test each print_epoch
       if np.mod(iepoch,print_epoch)==0:
          # apply the EMA parameters to evaluate
          # ema.apply_shadow()
          # set the model to eval for used in the model
          Prop_class.eval()
          # all_reduce the rmse form the training process 
          # here we dont need to recalculate the training error for saving the computation
          dist.all_reduce(lossprop, op=dist.ReduceOp.SUM)
          NC, NTA, lossE, lossG = lossprop
          loss = torch.tensor([torch.sqrt(lossE / NC), torch.sqrt(lossG / NTA)])
          
          # get the current rank and print the error in rank 0
          if rank==0:
              # lossprop=torch.sqrt(lossprop.detach().cpu())
              lr=optim.param_groups[0]["lr"]
              fout.write("{0:>4d}: {1:>8.2e} {2:5s} {3:>16.8f} {4:>16.8f}".format(iepoch,lr,' ',loss[0]*1000, loss[1]*1000))
              #fout.write('{:10.5f} '.format(loss[0]*1000))
              #fout.write('{:10.5f} '.format(loss[1]*1000))
              #for error in lossprop:
              #    fout.write('{:10.5f} '.format(error))
          
          # calculate the test error
          lossprop=torch.zeros(nprop+2,device=device)
          for data in data_test:
             abProp,cart,numatoms,species,atom_index,shifts=data
             loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts,\
             create_graph=False),abProp, numatoms=numatoms)
             
             lossprop[0] += len(numatoms)
             lossprop[1] += torch.sum(numatoms)
             lossprop[2] += loss_fn.lossE.item()
             lossprop[3] += loss_fn.lossG.item()

             # lossprop=lossprop+loss.detach()

          # all_reduce the rmse
          dist.all_reduce(lossprop.detach(),op=dist.ReduceOp.SUM)
          NC, NTA, lossE, lossG = lossprop
          loss = torch.tensor([torch.sqrt(lossE / NC), torch.sqrt(lossG / NTA)]).to(device=device)
          tot_loss = torch.sum(loss*prop_ceff[0:nprop])

          # loss=torch.sum(torch.mul(lossprop,prop_ceff[0:nprop]))
          # scheduler.step(tot_loss)
          lr=optim.param_groups[0]["lr"]
          # f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
          # prop_ceff[1]=f_ceff
          #  save the best model
          if tot_loss <best_loss[0]:
             best_loss[0]=tot_loss
             if rank == 0:
                 state = {'reannparam': Prop_class.state_dict(), 'optimizer': optim.state_dict()}
                 torch.save(state, "./REANN.pth")
                 PES_Normal.jit_pes()
                 if PES_Lammps:
                     PES_Lammps.jit_pes()
             # the barrier is used to prevent multiple processes from accessing the "REANN.pth" at the same time.
             # for example, when process 0  is saving the model to generate "REANN.pth" and the other processes are reading the "REANN.pth".
          
          # restore the model for continue training
          # ema.restore()
          # back to the best error
          #if tot_loss>25*best_loss[0] or tot_loss.isnan():
          #    dist.barrier()
          #    restart(Prop_class,"REANN.pth")
          #    # optim.param_groups[0]["lr"]=optim.param_groups[0]["lr"]*decay_factor
          #    # ema.restart()

          if rank==0:
              # lossprop=torch.sqrt(lossprop.detach().cpu()/test_nele)
              fout.write("{0:6s} {1:>16.8f} {2:>16.8f}".format(' ', loss[0]*1000, loss[1]*1000))
              # for error in lossprop:
              #    fout.write('{:10.5f} '.format(error))
              # if stop criterion
              fout.write("\n")
              fout.flush()
          if lr <=end_lr: break

