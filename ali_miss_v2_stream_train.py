import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
import argparse
import os
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import copy
import logging
from datetime import datetime
from torch.optim import lr_scheduler
import random

from trainers.ali_trainer_miss_v2 import AliMissStreamTrainer
from dataloaders.ali_odl_dataloader_miss_v2 import get_ali_stream_dataloader
from models.ali_pretrain_miss_v2 import PretrainMiss
from models.ali_stream_miss_v2 import StreamMiss
def parse_args():
    parser = argparse.ArgumentParser()
    # dataset 参数
    parser.add_argument('--data_path', type=str, default='./data/ali/processed_data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='./data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Ali', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=3, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=3, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=0.125, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=0.125, help='refund wait window size (days)')
    parser.add_argument('--stream_wait_window', type=int, default=0.25, help='stream wait window size (days)')
    # 预训练数据开始结束时间: 我们需要确定miss每一个头的正样本加权超参，这个统计值我们只能通过预训练的完整数据来获得
    parser.add_argument('--pretrain_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--pretrain_split_days_end', type=int, default=11, help='end day of train (days)')

    parser.add_argument('--train_split_days_start', type=int, default=17, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=24, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=17.25, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=24.25, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="miss_train_stream_v2", help='[miss_train_stream_v2]')
    # dataloader参数
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    # model参数
    parser.add_argument('--embed_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    # trainer参数
    parser.add_argument('--device_idx', type=str, default='1', help='Device index')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--model_save_pth', type=str, default='./pretrain_models/20250715', help='Model save pth')
    parser.add_argument('--pretrain_miss_model_pth', type=str, default='./pretrain_models/20250715/Ali_defer_pretrain.pth', help='if need a pretrain model to support')
    parser.add_argument('-reg_loss_decay',type=float,default=1e-4,help='Regularization loss decay coefficient')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha for miss')
    return parser.parse_args()

# 测试代码
if __name__ == '__main__':
    args = parse_args()

    train_stream_dataloader, test_stream_dataloader, miss_global_pos_weight = get_ali_stream_dataloader(args)

    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    pretrain_model = PretrainMiss(args).to(device)
    pretrain_model.load_state_dict(torch.load(args.pretrain_miss_model_pth))
    # 冷启动流模型
    stream_model = StreamMiss(args).to(device)
    # 加载部分层就行
    stream_model.load_state_dict(torch.load(args.pretrain_miss_model_pth),strict=False)


    stream_trainer = AliMissStreamTrainer(args, pretrain_model,stream_model,train_stream_dataloader,  test_stream_dataloader, miss_global_pos_weight)
    stream_trainer.train()

