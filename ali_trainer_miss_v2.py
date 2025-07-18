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
import torch.nn.functional as F
from mx_utils.metrics import auc_score,nll_score,prauc_score,pcoc_score



class AliPretrainMissTrainer(metaclass=ABCMeta):
    # 允许用真实的转化标签进行预训练
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s-%(message)s',
                            handlers=[logging.FileHandler(filename="./log/{}.txt".format(datetime.now().strftime("%Y-%m-%d"))),
                                      logging.StreamHandler()])
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,outputs,labels):
        return F.binary_cross_entropy(outputs, labels.float())

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        # 初始化
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_fn(outputs, pay_labels)
            loss.backward()
            self.optimizer.step()
            # 累加损失
            total_loss += loss.item()  # loss.item() 转换为 Python 浮动类型
            total_batches += 1  # 统计处理的批次数

            # metrics
            with torch.no_grad():
                outputs = self.model.predict(features)
                cvr_auc = auc_score(pay_labels, outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        # 计算并打印平均损失
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        # 更新学习率
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        # 每个epoch跑完存一下model
        # 这里简单做一个判断吧，后面再完善
        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        # 这里我只传了model 实例对象进来
        # 重新copy一下model，创建临时对象
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0

        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                outputs = Recmodel.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(outputs.cpu().numpy().tolist())

                # metrics
                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")

        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        # self.set_up_gpu(args)
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


    
class AliMissStreamTrainer(metaclass=ABCMeta):


        
    def __init__(self, args,pretrain_model,stream_model,train_stream_dataloader,test_stream_dataloader, miss_global_pos_weight):
        
        self.args=args
        self.setup_train(self.args)
        self.pretrain_model=pretrain_model
        self.pretrain_model.to(self.device)
        # 冻结所有参数
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

        self.stream_model=stream_model
        self.stream_model.to(self.device)

        self.train_stream_dataloader = train_stream_dataloader
        self.test_stream_dataloader = test_stream_dataloader

        # Miss的全局正样本加权权重
        self.miss_global_pos_weight = miss_global_pos_weight
        # Miss 调参权重
        self.alpha = args.alpha

        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode

        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")


        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)

        self.best_auc = 0

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s-%(message)s',
                            handlers=[logging.FileHandler(filename="./log/{}.txt".format(datetime.now().strftime("%Y-%m-%d"))),
                                      logging.StreamHandler()])
        self.logger= logging.getLogger(__name__)


    def loss_fn(self,head1_outputs,head2_outputs,head3_outputs,fused_outputs,stream_pay_labels,stream_pay_labels2,stream_pay_labels3,pos_weight,alpha,eps=1e-6):
        # Flatten
        head1_logits = head1_outputs.view(-1).clamp(min=eps, max=1-eps)
        head2_logits = head2_outputs.view(-1).clamp(min=eps, max=1-eps)
        head3_logits = head3_outputs.view(-1).clamp(min=eps, max=1-eps)
        fused_logits = fused_outputs.view(-1).clamp(min=eps, max=1-eps)

        # 1/2 pay_wait_window
        head1_pos_loss = -torch.log(head1_logits) * (1 + alpha * pos_weight[2])
        head1_neg_loss = -torch.log(1 - head1_logits)
        # 2/3 pay_wait_window
        head2_pos_loss = -torch.log(head2_logits) * (1 + alpha * pos_weight[1])
        head2_neg_loss = -torch.log(1 - head2_logits)
        # pay wait_window
        head3_pos_loss = -torch.log(head3_logits) * (1 + alpha * pos_weight[0])
        head3_neg_loss = -torch.log(1 - head3_logits)
        # fused
        fused_pos_loss = -torch.log(fused_logits) * (1 + alpha * pos_weight[0])
        fused_neg_loss = -torch.log(1 - fused_logits)

        # Weighted binary loss
        head1_loss = torch.mean(head1_pos_loss * stream_pay_labels2 + head1_neg_loss * (1 - stream_pay_labels2))
        head2_loss = torch.mean(head2_pos_loss * stream_pay_labels3 + head2_neg_loss * (1 - stream_pay_labels3))
        head3_loss = torch.mean(head3_pos_loss * stream_pay_labels + head3_neg_loss * (1 - stream_pay_labels))
        fused_loss = torch.mean(fused_pos_loss * stream_pay_labels + fused_neg_loss * (1 - stream_pay_labels))

        clf_loss = head1_loss + head2_loss + head3_loss + fused_loss


        return clf_loss
    

    


    def aggregate_metrics(self, metrics_list):
        # 初始化累计字典
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        # 累加
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        # 取平均
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        # 每天的数据单独训练
        for day in tqdm(range(len(self.train_stream_dataloader)), desc="Days"):
            # 每天的数据可以训练多个epoch
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  



        # 汇总所有天的指标
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        # 初始化
        total_loss = 0
        total_batches = 0


        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0


        train_stream_day_dataloader = self.train_stream_dataloader.get_day_dataloader(day_idx)

        tqdm_day_dataloader = tqdm(train_stream_day_dataloader, desc=f"Train Day {day_idx}", leave=False)

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            # 更小等待窗口的标签
            stream_pay_labels2 = batch['stream_pay_labels2'].to(self.device)
            stream_pay_labels3 = batch['stream_pay_labels3'].to(self.device)
            self.optimizer.zero_grad()
            head1_outputs,head2_outputs,head3_outputs,fused_outputs = self.stream_model(features)
            loss = self.loss_fn(head1_outputs,head2_outputs,head3_outputs,fused_outputs, stream_pay_labels,stream_pay_labels2,stream_pay_labels3,self.miss_global_pos_weight,self.alpha)
            loss.backward()
            self.optimizer.step()
            # 累加损失
            total_loss += loss.item()  # loss.item() 转换为 Python 浮动类型
            total_batches += 1  # 统计处理的批次数

            # metrics
            with torch.no_grad():
                head1_outputs,head2_outputs,head3_outputs,fused_outputs = self.stream_model.predict(features)
                # 用真实的标签去看一下效果
                cvr_auc = auc_score(pay_labels, fused_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,fused_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc
            tqdm_day_dataloader.set_description('Epoch {}, feature_loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))


        # 计算并打印平均损失
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Main Mean Loss: {mean_loss:.5f}")
        # 计算并打印平均指标
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        # 更新学习率
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        # 每个天的数据跑完存一下model
        # 这里简单做一个判断吧，后面再完善
        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")

        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        self.stream_model.eval()
        self.pretrain_model.eval()
    
        all_metrics = {}
        all_metrics["stream_model_CVR_AUC"] = 0
        all_metrics["stream_model_NetCVR_AUC"] = 0

        all_metrics["pretrained_model_CVR_AUC"] = 0
        all_metrics["pretrained_model_NetCVR_AUC"] = 0

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0

        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []
        with torch.no_grad():
            day_loader = self.test_stream_dataloader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                
                head1_outputs,head2_outputs,head3_outputs,fused_outputs = self.stream_model.predict(features)
                pretrained_outputs = self.pretrain_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy())
                stream_model_all_pay_preds.extend(fused_outputs.cpu().numpy())
                stream_model_all_net_pay_preds.extend(fused_outputs.cpu().numpy())
                pretrained_model_all_pay_preds.extend(pretrained_outputs.cpu().numpy())
                pretrained_model_all_net_pay_preds.extend(pretrained_outputs.cpu().numpy())
  
                # metrics
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, fused_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,fused_outputs)
                    all_metrics["stream_model_NetCVR_AUC"] += stream_model_net_cvr_auc

                    pretrained_model_cvr_auc = auc_score(pay_labels, pretrained_outputs)
                    all_metrics["pretrained_model_CVR_AUC"] += pretrained_model_cvr_auc

                    pretrained_model_net_cvr_auc = auc_score(net_pay_labels,pretrained_outputs)
                    all_metrics["pretrained_model_NetCVR_AUC"] += pretrained_model_net_cvr_auc

        all_metrics["stream_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["stream_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )




        self.logger.info(f"stream_model_CVR_AUC: {all_metrics['stream_model_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_NetCVR_AUC: {all_metrics['stream_model_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_CVR_AUC: {all_metrics['pretrained_model_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_NetCVR_AUC: {all_metrics['pretrained_model_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        # self.set_up_gpu(args)
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor(0.0, device=x.device)) + torch.log1p(torch.exp(-torch.abs(x)))


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset 参数
    parser.add_argument('--data_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/criteo/processed_data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Criteo', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=1, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=1, help='refund wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=15, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=15, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=30, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="defer_train_stream", help='[defer_pretrain,defer_dp_pretrain,defer_train_stream]')
    # dataloader参数
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    # model参数
    parser.add_argument('--embed_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    # trainer参数
    parser.add_argument('--device_idx', type=str, default='1', help='Device index')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--model_save_pth', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/pretrain_models/20250623', help='Model save pth')
    parser.add_argument('--pretrain_defer_dp_model_pth', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/pretrain_models/20250623/aux_model_pth1/dp_model.pth', help='if need a pretrain model1 to support')
    parser.add_argument('-reg_loss_decay',type=float,default=1e-4,help='Regularization loss decay coefficient')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

# 测试代码
if __name__ == '__main__':
    args = parse_args()



    # train_loader, test_loader = get_criteo_pretrain_defer_dataloader(args)
    # device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    # model = Defer(args).to(device)
    # defer_pretrain_trainer = CriteoPretrainDeferTrainer(args, model, train_loader, test_loader)
    # defer_pretrain_trainer.train()

    # for i, batch in enumerate(train_loader):
    #     print("i: ", i)
    #     print("batch: ", batch)
    #     print("batch['features'].shape: ", batch['features'].shape)

    # train_stream_dataloader, test_loader = get_criteo_defer_stream_dataloader(args)
    # for day in range(len(train_stream_dataloader)):
    #     print("day: ", day)
    #     day_loader = train_stream_dataloader.get_day_dataloader(day)
    #     for i, batch in enumerate(day_loader):
    #         print("i: ", i)
    #         print("batch: ", batch)