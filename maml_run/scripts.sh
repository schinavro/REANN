
for i in {Si Cu };
do 
mkdir $i
cd $i
python3 -m torch.distributed.run 

cd ..