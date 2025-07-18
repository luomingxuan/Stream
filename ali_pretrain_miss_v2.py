import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets_miss import num_bin_size,cate_bin_size

class PretrainMiss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.num_features = len(num_bin_size)
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(num_bin_size) + list(cate_bin_size)  # 所有桶大小
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])
        #  # 每列都映射成 8维，17列 → 136维
        input_dim = (self.num_features + self.cate_features) * args.embed_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)  # 输出1个logit

    def forward(self, x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)  # shape: [B, 17*8]
        x = F.leaky_relu(self.bn1(self.fc1(x_embed)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        logit = self.fc4(x)  # [B, 1]
        # shape: [B],输出一个转化概率
        return torch.sigmoid(logit).squeeze(1)
    

    def predict(self, x):
        """
        Args:
            x (Tensor): 输入张量，用于预测的数据。

        Returns:
            Tensor: 预测结果张量。

        """
        return self.forward(x)


