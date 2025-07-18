import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets_miss import num_bin_size,cate_bin_size

class StreamMiss(nn.Module):
    # 用于DFSN模型的冷启动
    def __init__(self,args):
        super().__init__()
        self.num_features = len(num_bin_size)
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(num_bin_size) + list(cate_bin_size)  # 所有桶大小
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])
        input_dim = (self.num_features + self.cate_features) * args.embed_dim

        # 迁移学习的线性权重
        self.w_m = nn.Parameter(torch.randn(input_dim, input_dim))
        self.w_f = nn.Parameter(torch.randn(input_dim, input_dim))
        self.w_u = nn.Parameter(torch.randn(input_dim, input_dim))

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        # 预训练的头，我们这里流训练的时候不要了
        self.fc4 = nn.Linear(128, 1)  # 输出1个logit

        self.head1 = nn.Linear(128, 1)
        self.head2 = nn.Linear(128, 1)
        self.head3 = nn.Linear(128, 1)
        # 因为有3 个head，[x_pred,x_norm]的拼接维度为6，生成的最后加权的维度为3
        self.fused_output_weight = nn.Linear(6,3)





    def forward (self, x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)  # shape: [B, 17*8]
        x = F.leaky_relu(self.bn1(self.fc1(x_embed)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        logit_head1 = torch.sigmoid(self.head1(x))  # [B, 1]
        logit_head2 = torch.sigmoid(self.head2(x))  # [B, 1]
        logit_head3 = torch.sigmoid(self.head3(x))  # [B, 1]

        x_pred = torch.cat([logit_head1, logit_head2, logit_head3], dim=-1)
        x_norm = torch.softmax(x_pred, dim=-1) 
        x_fused = torch.cat([x_pred, x_norm], dim=-1)
        x_fused = self.fused_output_weight(x_fused)
        x_fused = torch.softmax(x_fused, dim=-1)

        weighted_logit_head1 = x_fused[:,0] * logit_head1.squeeze(-1)
        weighted_logit_head2 = x_fused[:,1] * logit_head2.squeeze(-1)
        weighted_logit_head3 = x_fused[:,2] * logit_head3.squeeze(-1)

        fused_logit_out = weighted_logit_head1 + weighted_logit_head2 + weighted_logit_head3

        return logit_head1,logit_head2,logit_head3,fused_logit_out #[B]

    def predict(self,x):

        with torch.no_grad():
            return self.forward(x)



