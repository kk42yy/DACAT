conda activate pytorch1_13

export CUDA_VISIBLE_DEVICES=0

cd .../Cholec80/train_scripts

## Step 1
### train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1

## Step 2
## You need to use trained model in Step 1, and saved in .../train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT \
#     --epochs 100