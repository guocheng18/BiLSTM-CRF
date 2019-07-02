#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# default
nohup python -u train.py --name="default" > logs/default.log 2>&1 &

# dim (embed & hidden)
nohup python -u train.py --name="dimx2" --embed-dim=200 --hidden-dim=400 > logs/dimx2.log 2>&1 &
nohup python -u train.py --name="dimx3" --embed-dim=300 --hidden-dim=600 > logs/dimx3.log 2>&1 &

# lr
nohup python -u train.py --name="lr1e-2" --lr=1e-2 > logs/lr1e-2.log 2>&1 &
nohup python -u train.py --name="lr1e-4" --lr=1e-4 > logs/lr1e-4.log 2>&1 &

# dropout
nohup python -u train.py --name="dropout0.1" --dropout=0.1 > logs/dropout0.1.log 2>&1 &
nohup python -u train.py --name="dropout0.3" --dropout=0.3 > logs/dropout0.3.log 2>&1 &
nohup python -u train.py --name="dropout0.5" --dropout=0.5 > logs/dropout0.5.log 2>&1 &
