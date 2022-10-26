#!/bin/zsh

for i in Cu Ge Li Mo Ni Si
do 
    cd $i
    pwd
    python3 -m torch.distributed.run --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 48395 --nproc_per_node=2 --max_restarts=0 --standalone ../../../reann
    cd ..
done

