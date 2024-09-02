conda activate pytorch1_13

export CUDA_VISIBLE_DEVICES=0

cd .../AutoLaparo/train_scripts


python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
    --resume .../checkpoint_best_acc.pth.tar