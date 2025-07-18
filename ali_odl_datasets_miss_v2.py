# MISS 因为要处理三段流数据，不太好合并，我们重新写。
# MISS v2，作者没有公开代码，之前的做法是构成三段流数据，每个头分别训练。但是因为读入样本不一致，导致head之间难以形成相同的梯度收敛，因此换一种实现方法，这一次按照最大的等待窗口形成流，并在最大的等待窗口中标注不同等待窗口的转化标签
import shutil
import pickle
import pandas as pd
import argparse
import copy
import os
import numpy as np
import pickle
from tqdm import tqdm
import hashlib
from pandas.util import hash_pandas_object
SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
num_bin_size = ()
cate_bin_size = (8,32768,65536,65536,8192,65536,16,32,256,8192,16384,8,512,128,256,1024,128,16,8,8,16,16)

def get_data_df(args):
    print("Loading data from ", args.data_path)
    # 读取CSV文件，分隔符为制表符，没有表头
    df = pd.read_csv(args.data_path, sep="\t", header=None)
    print(df.head())
    print("preprocessing data from ",args.data_path)
    # 将第一列转换为numpy数组，表示点击时间戳
    click_ts = df[df.columns[0]].to_numpy()
    # 将第二列转换为numpy数组，缺失值填充为-1
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()
    refund_ts = df[df.columns[2]].fillna(-1).to_numpy()
    # -----------------------------------
    # 保留从第三列开始的所有列
    df = df[df.columns[3:]]
    # 将列名设置为0到23的字符串形式
    df.columns = [str(i) for i in range(22)]
    # 重置索引
    df.reset_index(drop=True)
    return df, click_ts, pay_ts, refund_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, refund_ts, sample_ts=None, pay_labels=None,net_pay_labels =None,refund_labels=None,stream_pay_labels=None,stream_pay_labels2=None,stream_pay_labels3=None,stream_net_pay_labels=None,
                 stream_pay_mask =None,delay_pay_labels_afterPay=None, delay_pay_label_afterRefund = None,delay_refund_label_afterRefund=None,inw_pay_labels_afterPay= None, inw_pay_labels_afterRefund=None, pay_attr_window = None, refund_attr_window = None):
        """
        初始化方法。
        Args:
            features (pandas.DataFrame): 特征数据，包含数值特征或ID特征。
            click_ts (numpy.ndarray): 点击时间戳数组。
            pay_ts (numpy.ndarray): 转化时间戳数组。
            refund_ts (numpy.ndarray): 退款时间戳数组。
            sample_ts (numpy.ndarray, optional): 采样时间戳数组，默认为 None。如果为 None，则使用点击时间戳作为采样时间戳。
            pay_labels (numpy.ndarray, optional): 真实的转化标签数组，默认为 None。如果为 None，则根据支付时间和点击时间生成转化标签。
            net_pay_labels (numpy.ndarray, optional): 真实的净转化标签数组（考虑转化和退款），默认为 None。如果为 None，则根据支付时间、退款时间和点击时间生成净转化标签。
            refund_labels (numpy.ndarray, optional): 真实的退款标签数组，默认为 None。如果为 None，则根据支付时间、退款时间和点击时间生成退款标签。
            stream_pay_labels (numpy.ndarray, optional): 流转化标签数组(主模型)，默认为 None。如果为 None，则使用转化标签作为流转化标签。
            stream_pay_labels2 (numpy.ndarray, optional): 流转化标签数组(次要模型)，默认为 None。如果为 None，则使用转化标签作为流转化标签。
            stream_pay_labels3 (numpy.ndarray, optional): 流转化标签数组(次要模型)，默认为 None。如果为 None，则使用转化标签作为流转化标签。
            stream_net_pay_labels (numpy.ndarray, optional): 流净转化标签数组(主模型)，默认为 None。如果为 None，则使用净转化标签作为流净转化标签。
            stream_pay_mask (numpy.ndarray, optional): 流转化标签掩码数组，默认为 None。如果为 None，则使用转化标签作为流转化标签掩码.用来区分是不是退款相关的标签
            delay_pay_labels_afterPay (numpy.ndarray, optional): 延迟转化的转化标签数组，默认为 None。如果为 None，则使用转化标签作为延迟转化的转化标签。
            delay_pay_label_afterRefund (numpy.ndarray, optional): 延迟退款的转化标签数组，默认为 None。如果为 None，则使用净转化标签作为延迟退款的转化标签。
            delay_refund_label_afterRefund (numpy.ndarray, optional): 延迟退款的退款标签数组，默认为 None。如果为 None，则使用退款标签作为延迟退款的退款标签。
            inw_pay_labels_afterPay (numpy.ndarray, optional): 在转化窗口内转化的转化标签数组，默认为 None。如果为 None，则使用转化标签作为在转化窗口内转化的转化标签。
            inw_pay_labels_afterRefund (numpy.ndarray, optional): 在退款窗口内考虑退款后的转化标签数组，默认为 None。如果为 None，则使用净转化标签作为在退款窗口内考虑退款后的转化标签。
            pay_attr_window (int, optional): 归因窗口（支付时间窗口），默认为 None。用于生成转化标签和净转化标签。
            refund_attr_window (int, optional): 归因窗口（退款时间窗口），默认为 None。用于生成退款标签。

        Returns:
            None

        """
        # feature 特征,这里是数值特征或者ID特征
        self.features = features.copy(deep=True)

        # 点击时间戳
        self.click_ts = copy.deepcopy(click_ts)
        # 转化时间戳
        self.pay_ts = copy.deepcopy(pay_ts)
        # 退款时间戳
        self.refund_ts = copy.deepcopy(refund_ts)
        # 延迟转化的转化标签
        self.delay_pay_labels_afterPay = delay_pay_labels_afterPay
        # 延迟退款的转化标签
        self.delay_pay_label_afterRefund = delay_pay_label_afterRefund
        # 在转化窗口内转化的转化标签
        self.inw_pay_labels_afterPay = inw_pay_labels_afterPay
        # 延迟退款的退款标签
        self.delay_refund_label_afterRefund = delay_refund_label_afterRefund
        # 在退款窗口内考虑退款后的转化标签
        self.inw_pay_labels_afterRefund = inw_pay_labels_afterRefund
        # 流转化标签
        self.stream_pay_labels = stream_pay_labels
        # 流转化标签
        self.stream_pay_labels2 = stream_pay_labels2
        # 流转化标签
        self.stream_pay_labels3 = stream_pay_labels3
        # 流净转化标签
        self.stream_net_pay_labels = stream_net_pay_labels

        # 流转化标签掩码
        self.stream_pay_mask = stream_pay_mask
        # 退款相关的窗口内退款标签
        # 设置采样时间
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        #  设置真实的转化标签（只考虑转化）
        if pay_labels is not None:
            # 如果传入了标签 labels，就直接使用（拷贝一份保存）。
            self.pay_labels = copy.deepcopy(pay_labels)
        else:
            # 如果设置了归因窗口的话， 用户确实支付了（pay_ts > 0）并且支付时间在点击后的归因窗口，判断有过支付并且能归因就行
            if pay_attr_window is not None:
                self.pay_labels = (np.logical_and(pay_ts > 0, pay_ts - click_ts < pay_attr_window)).astype(np.int32)
            # 没有设置归因窗口的话，只要有支付行为就算正样本。 
            else:
                self.pay_labels = (pay_ts > 0).astype(np.int32)
        
        # 设置真实的净转化标签(考虑转化何退款)
        if net_pay_labels is not None:
            self.net_pay_labels = copy.deepcopy(net_pay_labels)
        else:
            # 如果设置了归因窗口的话， 用户确实支付了（pay_ts > 0）并且支付时间在点击后的归因窗口，并且不能发生退款，或者退款超过最大的归因窗口（也就是退款窗口太长了就当作不退款了）
            if pay_attr_window is not None and refund_attr_window is not None:
                self.net_pay_labels = (
                    np.logical_and(
                        pay_ts > 0,
                        np.logical_and(
                            pay_ts - click_ts < pay_attr_window,
                            np.logical_or(
                                refund_ts < 0,
                                refund_ts - pay_ts >= refund_attr_window
                            )
                        )
                    )
                ).astype(np.int32)
            else:
                # 没有设置归因窗口的话，那么就是支付了并且没有退款
                self.net_pay_labels = (np.logical_and(pay_ts > 0, refund_ts < 0)).astype(np.int32)

        # 设置真实的退款标签
        if refund_labels is not None:
            self.refund_labels = copy.deepcopy(refund_labels)
        else:
            # 如果设置了归因窗口的话， 用户确实退款了（refund_ts > 0）并且退款时间在点击后的归因窗口，并且能发生退款
            if pay_attr_window is not None and refund_attr_window is not None:
                self.refund_labels = (
                    np.logical_and(
                        pay_ts > 0,
                            np.logical_and(
                                pay_ts - click_ts < pay_attr_window,
                                np.logical_and(
                                    refund_ts > 0,
                                    refund_ts - pay_ts < refund_attr_window
                                )
                        )
                    )
                ).astype(np.int32)
            else:
                self.refund_labels = (np.logical_and(pay_ts > 0, refund_ts > 0)).astype(np.int32)


        
        # 初始话延迟标签，全部改成pay_labels
        if self.delay_pay_labels_afterPay is None:
            self.delay_pay_labels_afterPay = self.pay_labels
        if self.inw_pay_labels_afterPay is None:
            self.inw_pay_labels_afterPay = self.pay_labels


        if self.delay_pay_label_afterRefund is None:
            self.delay_pay_label_afterRefund = self.net_pay_labels
        if self.delay_refund_label_afterRefund is None:
            self.delay_refund_label_afterRefund = self.refund_labels
        if self.inw_pay_labels_afterRefund is None:
            self.inw_pay_labels_afterRefund = self.net_pay_labels

        if self.stream_pay_labels is None:
            self.stream_pay_labels = self.pay_labels
        if self.stream_pay_labels2 is None:
            self.stream_pay_labels2 = self.pay_labels
        if self.stream_pay_labels3 is None:
            self.stream_pay_labels3 = self.pay_labels
        if self.stream_net_pay_labels is None:
            self.stream_net_pay_labels = self.net_pay_labels


        
        if self.stream_pay_mask is None:
            self.stream_pay_mask = np.ones_like(self.pay_labels)

        self.pay_attr_window = pay_attr_window
        self.refund_attr_window = refund_attr_window

    def sub_days(self,start_day,end_day):
        """
        提取指定日期范围内的数据。
        Args:
            start_day (int): 起始日期。
            end_day (int): 结束日期。
        Returns:
            DataDF: 返回一个新的DataDF对象，其中包含了指定日期范围内的数据。
        """
        # 将起始日期和结束日期转换为时间戳
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY 
        # 使用逻辑与操作筛选出时间戳在起始和结束日期之间的样本
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)

        # 根据筛选出的样本时间戳，构造并返回DataDF对象,注意这里的pay_labels,net_pay_labels,refund_labels都是真实标签，不考虑延迟的
        return DataDF(features = self.features.iloc[mask],
                click_ts = self.click_ts[mask],
                pay_ts = self.pay_ts[mask],
                refund_ts = self.refund_ts[mask],
                sample_ts = self.sample_ts[mask],
                pay_labels = self.pay_labels[mask],
                net_pay_labels = self.net_pay_labels[mask],
                refund_labels = self.refund_labels[mask],
                delay_pay_labels_afterPay = self.delay_pay_labels_afterPay[mask],
                delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[mask],
                inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                stream_pay_labels = self.stream_pay_labels[mask],
                pay_attr_window=self.pay_attr_window,
                refund_attr_window=self.refund_attr_window
                )

    def sub_days_v2(self, start_day, end_day, pay_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V2版本我们开始考虑实现转化延迟标签(不考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        # 将开始日期和结束日期转换为时间戳
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        # 创建一个掩码，用于筛选出位于开始日期和结束日期之间的样本
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        # 处理延迟转化标签（只考虑转化不考虑退款）
        # 如果attr_win不为空
        if self.pay_attr_window is not None:
            # 计算点击和支付之间的时间差
            diff = self.pay_ts - self.click_ts
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差在在观测窗口外，但是在归因窗口内的样本。这些样本就是作为我的延迟正样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
        else:
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差大于cut_size的样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
        
        # 复制转化标签以创建延迟标签
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        # 将不满足delay_mask条件的样本的延迟标签设置为0
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        # 返回处理后的数据框
        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                      inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)
    
    def sub_days_v3(self, start_day, end_day, pay_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V3版本我们开始考虑实现转化延迟标签,及时转化样本(不考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        # 将开始日期和结束日期转换为时间戳
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        # 创建一个掩码，用于筛选出位于开始日期和结束日期之间的样本
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        # 处理延迟转化标签（只考虑转化不考虑退款）
        # 如果attr_win不为空
        if self.pay_attr_window is not None:
            # 计算点击和支付之间的时间差
            diff = self.pay_ts - self.click_ts
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差在在观测窗口外，但是在归因窗口内的样本。这些样本就是作为我的延迟正样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差在在观测窗口内，但是在归因窗口内的样本。这些样本就是作为我的及时转化正样本
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff <= pay_wait_window, diff < self.pay_attr_window))
        else:
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差大于cut_size的样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= pay_wait_window)
       
        # 复制转化标签以创建延迟标签
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        # 将不满足delay_mask条件的样本的延迟标签设置为0
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        # 及时转化标签
        inw_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        # 将不满足immd_mask条件的样本的标签设置为0
        inw_pay_labels_afterPay[~inw_mask_afterPay] = 0

        # 返回处理后的数据框
        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                      inw_pay_labels_afterPay = inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def sub_days_v4(self, start_day, end_day, pay_wait_window,refund_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V4版本我们开始考虑实现转化延迟标签,及时转化样本(不考虑退款),延迟退款样本(考虑退款),及时退款样本(考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        # 将开始日期和结束日期转换为时间戳
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        # 创建一个掩码，用于筛选出位于开始日期和结束日期之间的样本
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        # 处理延迟转化标签（只考虑转化不考虑退款）
        # 如果attr_win不为空
        if self.pay_attr_window is not None:
            # 计算点击和支付之间的时间差
            diff = self.pay_ts - self.click_ts
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差在在观测窗口外，但是在归因窗口内的样本。这些样本就是作为我的延迟正样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差在在观测窗口内，但是在归因窗口内的样本。这些样本就是作为我的及时转化正样本
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff <= pay_wait_window, diff < self.pay_attr_window))
        else:
            # 创建一个掩码，用于筛选出支付时间大于0且点击和支付之间的时间差大于cut_size的样本
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= pay_wait_window)

        if self.refund_attr_window is not None:
            # 计算转化和退款之间的时间差
            diff = self.refund_ts - self.pay_ts
            # 创建一个掩码，用于筛选出退款时间大于0且转化和退款之间的时间差在在观测窗口外，但是在归因窗口内的样本。这些样本就是作为我的延迟退款正样本
            delay_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, np.logical_and(diff > refund_wait_window, diff < self.refund_attr_window)))
            # 创建一个掩码，用于筛选出退款时间大于0且转化和退款之间的时间差在在观测窗口内，但是在归因窗口内的样本。这些样本就是作为我的及时退款正样本
            inw_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, np.logical_and(diff <= refund_wait_window, diff < self.refund_attr_window)))
        else:
            # 创建一个掩码，用于筛选出退款时间大于0且转化和退款之间的时间差大于cut_size的样本
            delay_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, self.refund_ts - self.pay_ts > refund_wait_window))
            inw_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, self.refund_ts - self.pay_ts <= refund_wait_window))
       
        # 复制转化标签以创建延迟标签
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        # 将不满足delay_mask条件的样本的延迟标签设置为0
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        # 及时转化标签
        inw_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        # 将不满足immd_mask条件的样本的标签设置为0
        inw_pay_labels_afterPay[~inw_mask_afterPay] = 0

        # 复制退款标签以创建延迟标签
        delay_refund_labels_afterRefund = copy.deepcopy(self.refund_labels)
        # 将不满足delay_mask条件的样本的延迟标签设置为0
        delay_refund_labels_afterRefund[~delay_mask_afterRefund] = 0


        # 返回处理后的数据框
        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = delay_refund_labels_afterRefund[mask],
                      inw_pay_labels_afterPay = inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def get_global_pos_weights(self,pay_wait_window):
        """
        miss根据不同的转化等待窗口计算全局正样本权重,不考虑退款
        """
        pos_number = np.sum(self.pay_labels)
        delay_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window)
        delay_pos_number = np.sum(self.pay_labels[delay_pay_mask])
        return delay_pos_number / pos_number

    def add_miss_duplicate_samples(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款
        miss 有等待窗口，且回补延迟正样本

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        # 在观测窗口内转化的样本（可以立即观测到的正样本）
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window) # pay in wait_window
        # 采用更小的等待窗口
        inw_pay_mask2 = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window/2) # pay in wait_window
        # 采用更小的等待窗口
        inw_pay_mask3 = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window/3*2) # pay in wait_window

        delay_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window) # pay after wait_window

        # 观测数据，这个是特征
        df1 = self.features.copy(deep=True) # observe data
        # 复制数据，这里要复制全部的数据，所有真实的正样本（在归因窗口内发生转化）
        df2 = self.features.copy(deep=True) # 准备延迟副本
    
        # 重新组织一下数据，按照defer的逻辑的话这边相当于把原始样本复制两份
        new_features = pd.concat([
        df1[inw_pay_mask],          # A: 可观测正样本（立即支付）
        df1[~inw_pay_mask],         # B: 原始负样本 + 延迟转化
        df2[delay_pay_mask],        # C: 延迟支付延迟正样本回流
        ])
        # 采样时间
        sample_ts = np.concatenate([
            self.pay_ts[inw_pay_mask],      # A: 观测窗口内的正样本（立即支付）, 这里的采样时间因为考虑了延迟等待窗口，所以这里的采样时间直接设置成转化时间就行
            self.click_ts[~inw_pay_mask] + pay_wait_window,                       # B: 非观测窗口内的正样本（延迟支付） + 负样本 , 这里的采样时建也考虑到了延迟等待窗口，所以点击时间上也要加上等待窗口
            self.pay_ts[delay_pay_mask]                        # C: 对于所有真实的转化时间，在转化的时候再训练一次
        ])
        # 合并点击时间，把原始样本和延迟副本的点击时间拼接在一起，确保与后续的其他字段（如特征、支付时间、采样时间、标签等）保持对齐。
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask], \
            self.click_ts[delay_pay_mask]], axis=0)
        # 合并支付时间
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask], \
            self.pay_ts[delay_pay_mask]], axis=0)
        # 合并退款时间
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], \
                        self.refund_ts[delay_pay_mask]], axis=0)
    
        # 复制标签，这里是真实的转化标签
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        # 流转化标签插入延迟样本
        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),       # A: 观测窗口内支付 → 正样本
        np.zeros((np.sum(~inw_pay_mask),)),     # B: 非观测窗口支付 + 原始负样本 → 负样本
        pay_labels[delay_pay_mask],                 # C: 延迟支付副本 → 保持真实标签 1
        ], axis=0)

        # 更小的等待窗口的样本
        # 先复制一下原始样本
        pay_labels2 = copy.deepcopy(self.pay_labels)
        # 虽然在更大的等待窗口但要求标签不一样的设置
        pay_labels2[~inw_pay_mask2] = 0
        stream_pay_labels2 = np.concatenate([
        pay_labels2[inw_pay_mask],       # A: 观测窗口内支付 → 正样本
        np.zeros((np.sum(~inw_pay_mask),)),     # B: 非观测窗口支付 + 原始负样本 → 负样本
        pay_labels[delay_pay_mask],                 # C: 延迟支付副本 → 保持真实标签 1
        ], axis=0)

        # 更小的等待窗口的样本
        # 先复制一下原始样本
        pay_labels3 = copy.deepcopy(self.pay_labels)
        pay_labels3[~inw_pay_mask3] = 0
        stream_pay_labels3 = np.concatenate([
        pay_labels3[inw_pay_mask],       # A: 观测窗口内支付 → 正样本
        np.zeros((np.sum(~inw_pay_mask),)),     # B: 非观测窗口支付 + 原始负样本 → 负样本
        pay_labels[delay_pay_mask],                 # C: 延迟支付副本 → 保持真实标签 1
        ], axis=0)


        # 真实的转化标签
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask]                             
        ], axis=0)

        # 真实的净转化标签
        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask]          

        ], axis=0)

        # 真实的退款标签
        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask]          
        ], axis=0)

        # 排序索引
        # 这里做的是时间排序，让所有样本按照它们被“观察”的时间顺序排列：
        idx = list(range(new_features.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        # 这里的标签就是带有延迟标签的了（样本回流过的）
        return DataDF(features=new_features.iloc[idx],
                      click_ts=click_ts[idx],
                      pay_ts=pay_ts[idx],
                      sample_ts=sample_ts[idx],
                      refund_ts=refund_ts[idx],
                      pay_labels=pay_labels[idx],
                      net_pay_labels=net_pay_labels[idx],
                      refund_labels=refund_labels[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      stream_pay_labels2=stream_pay_labels2[idx],
                      stream_pay_labels3=stream_pay_labels3[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def shuffle(self):
        """
        打乱数据顺序。

        Args:
            无

        Returns:
            DataDF: 打乱顺序后的数据。

        """
        # 生成一个包含从0到x的行数范围的列表
        idx = list(range(self.features.shape[0]))
        # 随机打乱列表idx
        np.random.shuffle(idx)
        # 使用打乱后的索引idx重新排序self.x、self.click_ts、self.pay_ts、self.sample_ts、self.labels、self.delay_labels和self.inw_labels
        return DataDF(features = self.features.iloc[idx],
                      click_ts = self.click_ts[idx],
                      pay_ts = self.pay_ts[idx],
                      refund_ts = self.refund_ts[idx],
                      sample_ts = self.sample_ts[idx],
                      pay_labels = self.pay_labels[idx],
                      net_pay_labels = self.net_pay_labels[idx], 
                      refund_labels= self.refund_labels[idx],
                      delay_pay_labels_afterPay = self.delay_pay_labels_afterPay[idx],
                      inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[idx],
                      delay_pay_label_afterRefund=self.delay_pay_label_afterRefund[idx],
                      delay_refund_label_afterRefund=self.delay_refund_label_afterRefund[idx],
                      inw_pay_labels_afterRefund=self.inw_pay_labels_afterRefund[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

def get_ali_dataset_pretrain(args):
    np.random.seed(args.seed)
    dataset_name = args.dataset_name
    mode = args.mode
    print("Loading data from {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"{dataset_name}_{mode}.pkl")
    if os.path.isfile(cache_path):
        print("cache_path {} exists.".format(cache_path))
        print("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        print("building datasets")
        # 获取数据集特征，点击时间戳，转化时间戳，退款时间戳
        df,click_ts,pay_ts,refund_ts = get_data_df(args)
        pay_wait_window = args.pay_wait_window * SECONDS_A_DAY
        refund_wait_window = args.refund_wait_window * SECONDS_A_DAY
        pay_attr_window = args.pay_attr_window * SECONDS_A_DAY
        refund_attr_window = args.refund_attr_window * SECONDS_A_DAY
        data_src = DataDF(features = df, click_ts=click_ts, pay_ts=pay_ts, refund_ts=refund_ts,pay_attr_window=pay_attr_window,refund_attr_window=refund_attr_window)

        print("splitting into train and test sets")
        if mode == "miss_pretrain_v2":
            # miss预训练转化标签不考虑延迟，就是归因窗口内的转化标签
            # 因为miss 可以多头，但是我去做冷启的话统一是单头，用真实标签去训练。好处是每个头去预估的时候预训练都是无偏的。
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        else:
            raise ValueError(f"Unknown mode {mode}")
        print("writing data to cache file")
        if cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_data, "test": test_data}, f)
    print("data loaded successfully")
    print("====== TRAIN SET ======")
    print(f"Total samples                : {len(train_data.pay_labels):,}")
    print(f"Positive pay labels         : {sum(train_data.pay_labels):,}")
    print(f"Positive net pay labels   : {sum(train_data.net_pay_labels):,}")
    print(f"Positive refund labels      : {sum(train_data.refund_labels):,}")
    print(f"Positive delay pay labels   : {sum(train_data.delay_pay_labels_afterPay):,}")
    print(f"Positive delay refund labels   : {sum(train_data.delay_refund_label_afterRefund):,}")
    print("\n====== TEST SET ======")
    if hasattr(test_data, 'pay_labels'):
        print(f"Total samples                : {len(test_data.pay_labels):,}")
        print(f"Positive pay labels         : {sum(test_data.pay_labels):,}")
        print(f"Positive net pay labels   : {sum(test_data.net_pay_labels):,}")
        print(f"Positive refund labels      : {sum(test_data.refund_labels):,}")
        print(f"Positive delay pay labels   : {sum(test_data.delay_pay_labels_afterPay):,}")
        print(f"Positive delay refund labels   : {sum(test_data.delay_refund_label_afterRefund):,}")
    else:
        print(f"Total samples                : {len(test_data[0]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(test_data[0]['pay_labels']):,}")
        print(f"Positive net pay labels   : {sum(test_data[0]['net_pay_labels']):,}")
        print(f"Positive refund labels      : {sum(test_data[0]['refund_labels']):,}")
        print(f"Positive delay pay labels   : {sum(test_data[0]['delay_pay_labels_afterPay']):,}")
        print(f"Positive delay refund labels   : {sum(test_data[0]['delay_refund_label_afterRefund']):,}")


    return {
        "train": {
            "features": train_data.features,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "refund_ts": train_data.refund_ts,
            "pay_labels": train_data.pay_labels,
            "net_pay_labels": train_data.net_pay_labels,
            "refund_labels": train_data.refund_labels,
            "delay_pay_labels_afterPay" : train_data.delay_pay_labels_afterPay,
            "delay_pay_label_afterRefund" : train_data.delay_pay_label_afterRefund,
            "inw_pay_labels_afterPay" : train_data.inw_pay_labels_afterPay,
            "delay_refund_label_afterRefund" : train_data.delay_refund_label_afterRefund,
            "inw_pay_labels_afterRefund" : train_data.inw_pay_labels_afterRefund,
            "stream_pay_labels" : train_data.stream_pay_labels,
            "stream_net_pay_labels": train_data.stream_net_pay_labels,
            "stream_pay_mask": train_data.stream_pay_mask
        },
        "test": {
            "features": test_data.features,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": test_data.sample_ts,
            "refund_ts": test_data.refund_ts,
            "pay_labels": test_data.pay_labels,
            "net_pay_labels": test_data.net_pay_labels,
            "refund_labels": test_data.refund_labels,
            "delay_pay_labels_afterPay" : test_data.delay_pay_labels_afterPay,
            "delay_pay_label_afterRefund" : test_data.delay_pay_label_afterRefund,
            "inw_pay_labels_afterPay" : test_data.inw_pay_labels_afterPay,
            "delay_refund_label_afterRefund" : test_data.delay_refund_label_afterRefund,
            "inw_pay_labels_afterRefund" : test_data.inw_pay_labels_afterRefund,
            "stream_pay_labels" : test_data.stream_pay_labels,
            "stream_net_pay_labels": test_data.stream_net_pay_labels,
            "stream_pay_mask": test_data.stream_pay_mask
        }
    }


def get_ali_dataset_stream(args):
    """
    从指定路径加载或构建 Criteo 数据集，并返回训练和测试数据。

    Args:
        args (argparse.Namespace): 包含数据集名称、模式、数据缓存路径、等待窗口、属性窗口、训练分割起始和结束天数、测试分割起始和结束天数等参数。

    Returns:
        dict: 包含训练和测试数据的字典，其中训练数据以流的形式给出，测试数据以字典形式给出。

    """
    np.random.seed(args.seed)
    dataset_name = args.dataset_name
    mode = args.mode
    print("Loading data from {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"{dataset_name}_{mode}.pkl")
    if os.path.isfile(cache_path):
        print("cache_path {} exists.".format(cache_path))
        print("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_stream = data["train_stream"]
        test_stream = data["test_stream"]
        miss_global_pos_weight = data["miss_global_pos_weight"]
    
    else:
        print("building datasets")
        df,click_ts,pay_ts,refund_ts = get_data_df(args)
        pay_wait_window = args.pay_wait_window * SECONDS_A_DAY
        refund_wait_window = args.refund_wait_window * SECONDS_A_DAY
        pay_attr_window = args.pay_attr_window * SECONDS_A_DAY
        refund_attr_window = args.refund_attr_window * SECONDS_A_DAY
        data_src = DataDF(features = df, click_ts=click_ts, pay_ts=pay_ts, refund_ts=refund_ts,pay_attr_window=pay_attr_window,refund_attr_window=refund_attr_window)

        
        # pay_wait_window
        train_stream = []

        test_stream = []
        # miss 不同head 的全局正样本权重,head 我们设定死为3个，原文是5个，考虑计算开销我们直接删减
        # 1/2 * pay_wait_window , 2/3 * pay_wait_window, pay_wait_window
        miss_global_pos_weight =[0.1,0.1,0.1]
        print("splitting into train and test sets")

        if mode =="miss_train_stream_v2":
            # 需要通过预训练数据 得到pos 权重
            pretrain_data = data_src.sub_days(args.pretrain_split_days_start,args.pretrain_split_days_end)
            miss_global_pos_weight[0] = pretrain_data.get_global_pos_weights(pay_wait_window/2)
            miss_global_pos_weight[1] = pretrain_data.get_global_pos_weights(2*pay_wait_window/3)
            miss_global_pos_weight[2] = pretrain_data.get_global_pos_weights(pay_wait_window)

            # 得到miss的样本
            # 因为 miss 有不同的转化等待窗口，一定导致有不同的样本回补流（延迟转化样本不一样），因此后续模型训练的时候各个头肯定先单独训练，再联合训练。
            # 对于miss，因为是mutihead 模型，我们这里考虑开销问题，窗口大小问题直接设定死是3 head，然后不同head的等待窗口设置为 pay_wait_window, 2/3 * pay_wait_window, 1/2 * pay_wait_window
            train_data = data_src.sub_days(0, args.train_split_days_end).add_miss_duplicate_samples(pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)

            
            # 测试的话我们仍然去测真实的样本
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end) 
     
        else:
            raise ValueError(f"Unknown mode {mode}")
        # 按天进行分割stream,注意因为训练集天数固定，因此最后几天的数据的样本肯定回补补全，比如最后几天点击的样本但是还没到最大的归因窗口，因此不能回补
        for i in np.arange(args.train_split_days_start,args.train_split_days_end,args.stream_wait_window):
            train_data1_day = train_data.sub_days(i,i+args.stream_wait_window)
            train_stream.append({"features": train_data1_day.features,
                                    "click_ts": train_data1_day.click_ts,
                                    "pay_ts": train_data1_day.pay_ts,
                                    "sample_ts": train_data1_day.sample_ts,
                                    "refund_ts": train_data1_day.refund_ts,
                                    "pay_labels": train_data1_day.pay_labels,
                                    "net_pay_labels": train_data1_day.net_pay_labels,
                                    "refund_labels": train_data1_day.refund_labels,
                                    "delay_pay_labels_afterPay" : train_data1_day.delay_pay_labels_afterPay,
                                    "delay_pay_label_afterRefund" : train_data1_day.delay_pay_label_afterRefund,
                                    "inw_pay_labels_afterPay" : train_data1_day.inw_pay_labels_afterPay,
                                    "delay_refund_label_afterRefund" : train_data1_day.delay_refund_label_afterRefund,
                                    "inw_pay_labels_afterRefund" : train_data1_day.inw_pay_labels_afterRefund,
                                    "stream_pay_labels": train_data1_day.stream_pay_labels,
                                    "stream_pay_labels2": train_data1_day.stream_pay_labels2,
                                    "stream_pay_labels3": train_data1_day.stream_pay_labels3,
                                    "stream_net_pay_labels": train_data1_day.stream_net_pay_labels,
                                    "stream_pay_mask": train_data1_day.stream_pay_mask})


  
            
        for i in np.arange(args.test_split_days_start,args.test_split_days_end,args.stream_wait_window):
            test_day = test_data.sub_days(i,i+args.stream_wait_window)
            test_stream.append({"features": test_day.features,
                                "click_ts": test_day.click_ts,
                                "pay_ts": test_day.pay_ts,
                                "sample_ts": test_day.sample_ts,
                                "refund_ts": test_day.refund_ts,
                                "pay_labels": test_day.pay_labels,
                                "net_pay_labels": test_day.net_pay_labels,
                                "refund_labels": test_day.refund_labels,
                                "delay_pay_labels_afterPay" : test_day.delay_pay_labels_afterPay,
                                "delay_pay_label_afterRefund" : test_day.delay_pay_label_afterRefund,
                                "inw_pay_labels_afterPay" : test_day.inw_pay_labels_afterPay,
                                "delay_refund_label_afterRefund": test_day.delay_refund_label_afterRefund,
                                "inw_pay_labels_afterRefund": test_day.inw_pay_labels_afterRefund,
                                "stream_pay_labels": test_day.stream_pay_labels,
                                "stream_pay_labels2": test_day.stream_pay_labels2,
                                "stream_pay_labels3": test_day.stream_pay_labels3,
                                "stream_net_pay_labels": test_day.stream_net_pay_labels,
                                "stream_pay_mask": test_day.stream_pay_mask})
            
        print("writing data to cache file")
        if cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train_stream": train_stream,"test_stream": test_stream,"miss_global_pos_weight":miss_global_pos_weight}, f)

    print("====== Train SET ======")
    for day in range(len(train_stream)):
        print("Day",day)
        print(f"Total samples                : {len(train_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(train_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(train_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(train_stream[day]['refund_labels']):,}")
    print("====== Test SET ======")
    for day in range(len(test_stream)):
        print("Day",day)
        print(f"Total samples                : {len(test_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(test_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(test_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(test_stream[day]['refund_labels']):,}")
    print("miss_global_pos_weight",miss_global_pos_weight)
    return {"train_stream": train_stream,"test_stream": test_stream,"miss_global_pos_weight":miss_global_pos_weight}
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/criteo/processed_data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Criteo', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=1, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=1, help='refund wait window size (days)')
    parser.add_argument('--stream_wait_window', type=int, default=1, help='stream wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=15, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=15, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=30, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="miss_pretrain_v2", help='[miss_pretrain_v2, miss_train_stream]')
    return parser.parse_args()

# 测试代码
if __name__ == '__main__':
    args = parse_args()

    data_src = get_ali_dataset_stream(args)
    train_stream = data_src['train']
    for i in range(len(train_stream)):
        print(i)
        print('pos samples',sum(train_stream[i]['pay_labels']))
        print('total samples',len(train_stream[i]['pay_labels']))
        print('delay pos samples',sum(train_stream[i]['delay_pay_labels_afterPay']))

