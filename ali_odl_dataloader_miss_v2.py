# dfsn的dataloader我们也需要重写，因为他需要三个不同的流
from torch.utils.data import Dataset, DataLoader
import torch
import shutil
import pickle
import pandas as pd
import argparse
import copy
import os
import numpy as np
import pickle

from datasets.ali_odl_datasets_miss_v2 import get_ali_dataset_pretrain,get_ali_dataset_stream



class AliPretrainOdlDatasets(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict: 来自 get_criteo_dataset_pretrain() 返回的 'train' 或 'test' 字典
        """
        # 特征：DataFrame -> NumPy -> Tensor
        self.features = torch.tensor(data_dict["features"].to_numpy(), dtype=torch.float32)
        self.pay_labels = torch.tensor(data_dict["pay_labels"], dtype=torch.float32)
        self.net_pay_labels = torch.tensor(data_dict["net_pay_labels"], dtype=torch.float32)
        self.refund_labels = torch.tensor(data_dict["refund_labels"], dtype=torch.float32)
        self.delay_pay_labels_afterPay = torch.tensor(data_dict["delay_pay_labels_afterPay"], dtype=torch.float32)
        self.delay_pay_label_afterRefund = torch.tensor(data_dict["delay_pay_label_afterRefund"], dtype=torch.float32)
        self.inw_pay_labels_afterPay = torch.tensor(data_dict["inw_pay_labels_afterPay"], dtype=torch.float32)
        self.delay_refund_label_afterRefund = torch.tensor(data_dict["delay_refund_label_afterRefund"], dtype=torch.float32)
        self.inw_pay_labels_afterRefund = torch.tensor(data_dict["inw_pay_labels_afterRefund"], dtype=torch.float32)
        self.stream_pay_labels = torch.tensor(data_dict["stream_pay_labels"], dtype=torch.float32)
        self.stream_net_pay_labels = torch.tensor(data_dict["stream_net_pay_labels"], dtype=torch.float32)
        self.stream_pay_mask = torch.tensor(data_dict["stream_pay_mask"], dtype=torch.float32)
        self.click_ts = torch.tensor(data_dict["click_ts"], dtype=torch.float32)
        self.pay_ts = torch.tensor(data_dict["pay_ts"], dtype=torch.float32)
        self.refund_ts = torch.tensor(data_dict["refund_ts"], dtype=torch.float32)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],  # shape: (M,)
            "pay_labels": self.pay_labels[idx],
            "net_pay_labels": self.net_pay_labels[idx],
            "refund_labels": self.refund_labels[idx],
            "delay_pay_labels_afterPay": self.delay_pay_labels_afterPay[idx],
            "delay_pay_label_afterRefund": self.delay_pay_label_afterRefund[idx],
            "inw_pay_labels_afterPay": self.inw_pay_labels_afterPay[idx],
            "delay_refund_label_afterRefund": self.delay_refund_label_afterRefund[idx],
            "inw_pay_labels_afterRefund": self.inw_pay_labels_afterRefund[idx],
            "click_ts": self.click_ts[idx],
            "pay_ts": self.pay_ts[idx],
            "refund_ts": self.refund_ts[idx],
        }
    
class AliOdlStreamDataset(Dataset):
    # dfsn 主塔的dataset 数据
    def __init__(self, data_dict):
        """
        data_dict: 单天的训练数据字典train_stream[i]
        label_key: 使用哪个标签字段（如 'net_pay_labels', 'delay_pay_labels_afterPay'）
        """
        # 特征：DataFrame -> NumPy -> Tensor
        self.features = torch.tensor(data_dict["features"].to_numpy(), dtype=torch.float32)
        self.pay_labels = torch.tensor(data_dict["pay_labels"], dtype=torch.float32)
        self.net_pay_labels = torch.tensor(data_dict["net_pay_labels"], dtype=torch.float32)
        self.refund_labels = torch.tensor(data_dict["refund_labels"], dtype=torch.float32)
        self.delay_pay_labels_afterPay = torch.tensor(data_dict["delay_pay_labels_afterPay"], dtype=torch.float32)
        self.delay_pay_label_afterRefund = torch.tensor(data_dict["delay_pay_label_afterRefund"], dtype=torch.float32)
        self.inw_pay_labels_afterPay = torch.tensor(data_dict["inw_pay_labels_afterPay"], dtype=torch.float32)
        self.delay_refund_label_afterRefund = torch.tensor(data_dict["delay_refund_label_afterRefund"], dtype=torch.float32)
        self.inw_pay_labels_afterRefund= torch.tensor(data_dict["inw_pay_labels_afterRefund"], dtype=torch.float32)
        self.stream_pay_labels = torch.tensor(data_dict["stream_pay_labels"], dtype=torch.float32)
        self.stream_pay_labels2 = torch.tensor(data_dict["stream_pay_labels2"], dtype=torch.float32)
        self.stream_pay_labels3 = torch.tensor(data_dict["stream_pay_labels3"], dtype=torch.float32)
        self.stream_net_pay_labels = torch.tensor(data_dict["stream_net_pay_labels"], dtype=torch.float32)
        self.stream_pay_mask = torch.tensor(data_dict["stream_pay_mask"], dtype=torch.float32)
        self.click_ts = torch.tensor(data_dict["click_ts"], dtype=torch.float32)
        self.pay_ts = torch.tensor(data_dict["pay_ts"], dtype=torch.float32)
        self.refund_ts = torch.tensor(data_dict["refund_ts"], dtype=torch.float32)
    def __len__(self):
        return len(self.pay_labels)
    def __getitem__(self, idx):
        return {
            "features": self.features[idx],  # shape: (M,)
            "pay_labels": self.pay_labels[idx],
            "net_pay_labels": self.net_pay_labels[idx],
            "refund_labels": self.refund_labels[idx],
            "delay_pay_labels_afterPay": self.delay_pay_labels_afterPay[idx],
            "delay_pay_label_afterRefund": self.delay_pay_label_afterRefund[idx],
            "inw_pay_labels_afterPay": self.inw_pay_labels_afterPay[idx],
            "delay_refund_label_afterRefund": self.delay_refund_label_afterRefund[idx],
            "inw_pay_labels_afterRefund": self.inw_pay_labels_afterRefund[idx],
            "stream_pay_labels": self.stream_pay_labels[idx],
            "stream_pay_labels2": self.stream_pay_labels2[idx],
            "stream_pay_labels3": self.stream_pay_labels3[idx],
            "stream_net_pay_labels": self.stream_net_pay_labels[idx],
            "stream_pay_mask": self.stream_pay_mask[idx],
            "click_ts": self.click_ts[idx],
            "pay_ts": self.pay_ts[idx],
            "refund_ts": self.refund_ts[idx],
        }




class AliOdlStreamLoader:
    def __init__(self, stream_data, batch_size=1024, shuffle=False, num_workers=4):

        # 初始化CriteoDeferStreamLoader类，传入stream_data、batch_size、shuffle和num_workers参数
        self.stream_data = stream_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __len__(self):
        # 返回stream_data的长度
        return len(self.stream_data)

    def get_day_dataloader(self, day):
        # 根据传入的day参数，获取对应天的数据集，并返回DataLoader
        dataset = AliOdlStreamDataset(self.stream_data[day])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

def get_ali_pretrain_dataloader(args):
    """
    获取ali预训练加载的数据加载器。

    Args:
        args (argparse.Namespace): 包含配置参数的命名空间对象。

    Returns:
        tuple: 包含训练集和测试集数据加载器的元组。
            - train_dataloader (DataLoader): 训练集数据加载器。
            - test_dataloader (DataLoader): 测试集数据加载器。

    """


    # 注意此时根据mode得到返回的标签到底是否为延迟转化标签
    data_src = get_ali_dataset_pretrain(args)
    train_data = data_src['train']
    test_data = data_src['test']
    train_dataset = AliPretrainOdlDatasets(train_data)
    test_dataset = AliPretrainOdlDatasets(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, test_dataloader

def get_ali_stream_dataloader(args):
    """
    根据传入的参数创建一个Criteo的延迟转化流数据加载器。

    Args:
        args (argparse.Namespace): 包含训练参数的对象，如批处理大小、工作线程数等。

    Returns:
        tuple: 包含两个元素，分别是训练数据加载器和测试数据加载器。

            - train_stream_dataloader (CriteoDeferStreamLoader): 用于加载训练数据的延迟转化流数据加载器。
            - test_dataloader (DataLoader): 用于加载测试数据的普通数据加载器。

    """
    # 注意此时根据mode得到返回的标签到底是否为延迟转化标签
    data_src = get_ali_dataset_stream(args)
    train_stream = data_src['train_stream']
    test_stream = data_src['test_stream']
    miss_global_pos_weight = data_src['miss_global_pos_weight']

    train_stream_dataloader = AliOdlStreamLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_stream_dataloader = AliOdlStreamLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_stream_dataloader,  test_stream_dataloader, miss_global_pos_weight



